"""
PAN-WVC-10 Corpus Manager for GraphRAG Poisoning Experiments.

Loads the PAN Wikipedia Vandalism Corpus 2010, builds a "start" subset of
initial article versions, and applies clean or vandalism changes from the
corpus to documents in that subset.

Expected directory layout (after extracting pan-wikipedia-vandalism-corpus-2010.zip):

    pan-wikipedia-vandalism-corpus-2010/
        edits.csv              # edit metadata
        gold-annotations.csv   # labels (class: "yes" / "no")
        article-revisions/     # revision text files
            partXX/
                <revision_id>.txt
                ...
"""

from __future__ import annotations

import difflib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _strip_wikitext(text: str) -> str:
    import mwparserfromhell
    return mwparserfromhell.parse(text).strip_code(normalize=True, collapse=True).strip()

import pandas as pd


CleanOrVandalism = Literal["clean", "vandalism"]


def _truncate_words(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[:n_words])


# --------------------------------------------------------------------------- #
# Data model                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class ChangeRecord:
    """Metadata about a single change applied to a document."""

    editid: str
    article_title: str
    filename: str
    old_revision_id: str
    new_revision_id: str
    old_content: str
    new_content: str
    label: Literal["yes", "no"]              # "yes" = vandalism, "no" = regular
    is_vandalism: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_vandalism = self.label == "yes"


# --------------------------------------------------------------------------- #
# Corpus manager                                                              #
# --------------------------------------------------------------------------- #

class PANWVCCorpus:
    """Loads PAN-WVC-10 and provides start-set / apply-change operations."""

    def __init__(self, corpus_path: str | Path, seed: int = 42) -> None:
        self.corpus_path = Path(corpus_path)
        self.revisions_dir = self._find_revisions_dir()
        self.edits: pd.DataFrame = self._load_metadata()
        self._rng = random.Random(seed)
        self._revision_index: dict[str, Path] = {
            p.stem: p for p in self.revisions_dir.rglob("*.txt")
        }
        cache = self.corpus_path / "categories.csv"
        self._category_index: pd.DataFrame | None = pd.read_csv(cache) if cache.exists() else None

    # ------------------------------------------------------------------- #
    # Loading                                                             #
    # ------------------------------------------------------------------- #

    def _find_revisions_dir(self) -> Path:
        for name in ("article-revisions", "revisions"):
            candidate = self.corpus_path / name
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"No article-revisions/ or revisions/ subdir in {self.corpus_path}"
        )

    def _load_metadata(self) -> pd.DataFrame:
        """Load edits + gold-annotations and return a DataFrame containing 'class'."""
        edits = pd.read_csv(self.corpus_path / "edits.csv")
        if "class" in edits.columns:
            return edits  # already merged
        annotations = pd.read_csv(self.corpus_path / "gold-annotations.csv")
        return edits.merge(annotations, on="editid", how="inner")

    def build_category_index(self, cache_path: str | Path | None = None) -> pd.DataFrame:
        """Fetch Wikipedia categories for every article and cache to CSV.

        Requires ``requests``. Safe to call multiple times — returns the cached
        DataFrame immediately if the cache file already exists.

        Args:
            cache_path: Where to write the CSV. Defaults to
                ``<corpus_path>/categories.csv``.

        Returns:
            DataFrame with columns ``articletitle`` and ``category``.
        """
        import requests
        import time

        if cache_path is None:
            cache_path = self.corpus_path / "categories.csv"
        cache_path = Path(cache_path)

        if cache_path.exists():
            self._category_index = pd.read_csv(cache_path)
            return self._category_index

        titles = self.edits["articletitle"].unique().tolist()
        rows: list[dict] = []
        batch_size = 50
        n_batches = (len(titles) + batch_size - 1) // batch_size

        session = requests.Session()
        session.headers["User-Agent"] = "GraphPoison-research/1.0"

        for i in range(0, len(titles), batch_size):
            batch_num = i // batch_size
            batch = titles[i : i + batch_size]
            params = {
                "action": "query",
                "prop": "categories",
                "titles": "|".join(batch),
                "cllimit": "max",
                "clshow": "!hidden",
                "format": "json",
            }
            retries = 3
            for attempt in range(retries):
                try:
                    resp = session.get(
                        "https://en.wikipedia.org/w/api.php",
                        params=params,
                        timeout=30,
                    )
                    if resp.status_code == 429:
                        wait = int(resp.headers.get("Retry-After", 5)) + 1
                        print(f"[rate-limit] batch {batch_num}/{n_batches} — waiting {wait}s")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    pages = resp.json().get("query", {}).get("pages", {})
                    for page in pages.values():
                        title = page.get("title", "")
                        for cat in page.get("categories", []):
                            rows.append({"articletitle": title, "category": cat["title"]})
                    break
                except Exception as exc:
                    print(f"[warn] batch {batch_num}/{n_batches} attempt {attempt+1}: {exc}")
                    time.sleep(2)
            else:
                print(f"[skip] batch {batch_num}/{n_batches} failed after {retries} attempts")

            print(f"[progress] {batch_num+1}/{n_batches} batches done", end="\r")
            time.sleep(1)  # stay well under Wikipedia's rate limit

        df = pd.DataFrame(rows, columns=["articletitle", "category"])
        df.to_csv(cache_path, index=False)
        self._category_index = df
        print(f"[ok] Category index written -> {cache_path}")
        return df

    def list_categories(self) -> list[str]:
        """Return all unique category names present in the index."""
        if self._category_index is None:
            raise RuntimeError("Category index not built. Call build_category_index() first.")
        return sorted(self._category_index["category"].unique().tolist())

    def _load_revision_text(self, revision_id: str | int) -> str:
        rev_id = str(revision_id)
        path = self._revision_index.get(rev_id)
        if path is None:
            raise FileNotFoundError(f"Revision text not found: {rev_id}")
        return path.read_text(encoding="utf-8", errors="replace")

    # ------------------------------------------------------------------- #
    # Public API                                                          #
    # ------------------------------------------------------------------- #

    def build_corpus_jsonl(
        self,
        output_path: str | Path,
        subset_size: int | None = None,
        categories: list[str] | None = None,
        category_pattern: str | None = None,
    ) -> int:
        """Write articles as corpus.jsonl compatible with LogicPoison.

        Each line: {"_id": "<n>", "title": "<articletitle>", "text": "<content>"}

        Args:
            output_path: Path to the output corpus.jsonl file.
            subset_size: If given, cap to this many articles.
            categories: If given, restrict to articles in exactly these Wikipedia categories.
            category_pattern: If given, restrict to articles in any category whose name matches
                this regex (case-insensitive) - e.g. "footballer|association football" to pull in
                a whole topically-coherent cluster (by position/nationality/club/etc.) without
                listing every one of its hundreds of sub-category names by hand. Takes precedence
                over `categories` if both are given.

        Returns:
            Number of records written.
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        per_article = (
            self.edits
            .sort_values("edittime")
            .groupby("articletitle", as_index=False)
            .first()
        )

        if categories is not None or category_pattern is not None:
            if self._category_index is None:
                raise RuntimeError("Category index not built. Call build_category_index() first.")
            if category_pattern is not None:
                cat_mask = self._category_index["category"].str.contains(category_pattern, case=False, regex=True, na=False)
            else:
                cat_mask = self._category_index["category"].isin(categories)
            matching_titles = self._category_index[cat_mask]["articletitle"].unique()
            per_article = per_article[per_article["articletitle"].isin(matching_titles)]

        if subset_size is not None and subset_size < len(per_article):
            per_article = per_article.sample(
                n=subset_size,
                random_state=self._rng.randint(0, 2**32 - 1),
            )

        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for idx, (_, row) in enumerate(per_article.iterrows()):
                try:
                    content = self._load_revision_text(row["oldrevisionid"])
                except FileNotFoundError:
                    continue
                record = {"_id": str(idx), "title": str(row["articletitle"]), "text": _strip_wikitext(content)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return count

    def apply_change(
        self,
        start_dir: str | Path,
        change_type: CleanOrVandalism,
    ) -> ChangeRecord:
        """Apply a clean or vandalism change to a random document in `start_dir`.

        Picks a random edit of the requested type whose article exists in
        `start_dir`, replaces the document content with the new revision text,
        and returns metadata about what was changed.

        Args:
            start_dir: Directory containing the current state of documents.
            change_type: "clean" or "vandalism".

        Returns:
            A `ChangeRecord` describing the applied change.
        """
        start_dir = Path(start_dir)
        label: Literal["yes", "no"] = "yes" if change_type == "vandalism" else "no"

        pool = self.edits[self.edits["class"] == label]
        if pool.empty:
            raise ValueError(f"No edits with class={label!r} in corpus")

        # try edits in random order until one matches a doc in start_dir
        shuffled = pool.sample(
            frac=1.0, random_state=self._rng.randint(0, 2**32 - 1)
        )
        for _, row in shuffled.iterrows():
            fname = self._sanitize(row["articletitle"]) + ".txt"
            target = start_dir / fname
            if not target.exists():
                continue
            try:
                new_content = self._load_revision_text(row["newrevisionid"])
            except FileNotFoundError:
                continue
            old_content = target.read_text(encoding="utf-8", errors="replace")
            target.write_text(new_content, encoding="utf-8")
            return ChangeRecord(
                editid=str(row["editid"]),
                article_title=str(row["articletitle"]),
                filename=fname,
                old_revision_id=str(row["oldrevisionid"]),
                new_revision_id=str(row["newrevisionid"]),
                old_content=old_content,
                new_content=new_content,
                label=label,
            )

        raise RuntimeError(
            f"No applicable {change_type} edit found for documents in {start_dir}"
        )

    def get_diffs(
        self,
        change_type: CleanOrVandalism | None = None,
        limit: int | None = None,
        document_length: int | None = None,
        padding_lines: int = 3,
        categories: list[str] | None = None,
        article_titles: set[str] | None = None,
    ) -> list[dict]:
        """Return diffs as ``{"original_text": ..., "new_text": ..., ...}`` dicts.

        When ``document_length`` is given:
          1. Documents already within ±5 % of that length are preferred and
             returned as-is.
          2. If fewer than ``limit`` such documents exist, remaining slots are
             filled by cutting other documents to ``document_length`` characters,
             extracting the window that contains the diff (first changed line →
             last changed line, plus ``padding_lines`` on each side).

        Args:
            change_type: ``"clean"`` for regular edits, ``"vandalism"`` for
                vandalism edits, or ``None`` for both.
            limit: Maximum number of diffs to return.
            document_length: Target character length for returned texts.
            padding_lines: Lines of context to keep before/after the changed
                region when cutting a document to size.

        Returns:
            List of dicts with keys:
                original_text, new_text, editid, article_title,
                old_revision_id, new_revision_id, is_vandalism
        """
        CLASS_MAP: dict[CleanOrVandalism, str] = {
            "clean": "regular",
            "vandalism": "vandalism",
        }

        if change_type is not None:
            pool = self.edits[self.edits["class"] == CLASS_MAP[change_type]]
        else:
            pool = self.edits[self.edits["class"].isin(CLASS_MAP.values())]

        if categories is not None:
            if self._category_index is None:
                raise RuntimeError("Category index not built. Call build_category_index() first.")
            matching_titles = self._category_index[
                self._category_index["category"].isin(categories)
            ]["articletitle"].unique()
            pool = pool[pool["articletitle"].isin(matching_titles)]

        if article_titles is not None:
            pool = pool[pool["articletitle"].isin(article_titles)]

        # Shuffle once so both passes draw from the same random order.
        pool = pool.sample(frac=1.0, random_state=self._rng.randint(0, 2**32 - 1))

        if document_length is None:
            # Original behaviour: no length constraint.
            rows = pool if limit is None else pool.head(limit)
            diffs: list[dict] = []
            for _, row in rows.iterrows():
                try:
                    original_text = self._load_revision_text(row["oldrevisionid"])
                    new_text = self._load_revision_text(row["newrevisionid"])
                except FileNotFoundError:
                    continue
                diffs.append(self._make_diff_dict(row, original_text, new_text))
            return diffs

        lo = int(document_length * 0.95)
        hi = int(document_length * 1.05)

        right_size: list[dict] = []
        need_cutting: list[tuple] = []  # (row, original_text, new_text)

        for _, row in pool.iterrows():
            if limit is not None and len(right_size) >= limit:
                break
            try:
                original_text = self._load_revision_text(row["oldrevisionid"])
                new_text = self._load_revision_text(row["newrevisionid"])
            except FileNotFoundError:
                continue
            if lo <= len(new_text.split()) <= hi:
                right_size.append(self._make_diff_dict(row, original_text, new_text))
            else:
                need_cutting.append((row, original_text, new_text))

        if limit is not None and len(right_size) >= limit:
            return right_size[:limit]

        # Fill remaining slots by cutting documents to size.
        result = list(right_size)
        remaining = (limit - len(result)) if limit is not None else len(need_cutting)
        for row, original_text, new_text in need_cutting[:remaining]:
            orig_chunk, new_chunk = self._extract_chunk_around_diff(
                original_text, new_text, document_length, padding_lines
            )
            result.append(self._make_diff_dict(row, orig_chunk, new_chunk))

        return result

    def _make_diff_dict(self, row: "pd.Series", original_text: str, new_text: str) -> dict:
        return {
            "original_text": original_text,
            "new_text": new_text,
            "editid": str(row["editid"]),
            "article_title": str(row["articletitle"]),
            "old_revision_id": str(row["oldrevisionid"]),
            "new_revision_id": str(row["newrevisionid"]),
            "is_vandalism": row["class"] == "vandalism",
        }

    def _extract_chunk_around_diff(
        self,
        original: str,
        new: str,
        length: int,
        padding_lines: int = 3,
    ) -> tuple[str, str]:
        """Return ``(orig_chunk, new_chunk)`` of ≤ ``length`` words centred on the diff."""
        orig_lines = original.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        opcodes = difflib.SequenceMatcher(None, orig_lines, new_lines, autojunk=False).get_opcodes()

        changed_orig = [i for tag, i1, i2, _, _ in opcodes if tag != "equal" for i in range(i1, i2)]
        changed_new  = [j for tag, _, _, j1, j2 in opcodes if tag != "equal" for j in range(j1, j2)]

        if not changed_new and not changed_orig:
            return _truncate_words(original, length), _truncate_words(new, length)

        orig_start = max(0, min(changed_orig, default=0) - padding_lines)
        orig_end   = min(len(orig_lines), max(changed_orig, default=len(orig_lines)) + padding_lines + 1)
        new_start  = max(0, min(changed_new, default=0) - padding_lines)
        new_end    = min(len(new_lines),  max(changed_new, default=len(new_lines))  + padding_lines + 1)

        orig_chunk = _truncate_words("".join(orig_lines[orig_start:orig_end]), length)
        new_chunk  = _truncate_words("".join(new_lines[new_start:new_end]), length)

        return orig_chunk, new_chunk

    # ------------------------------------------------------------------- #
    # Helpers                                                             #
    # ------------------------------------------------------------------- #

    @staticmethod
    def _sanitize(title: str) -> str:
        """Make an article title safe for use as a filename."""
        keep = "-_.()"
        cleaned = "".join(
            c if c.isalnum() or c in keep else "_" for c in str(title)
        )
        return cleaned.strip("_")
    
    # creating a queries.jsonl file for a given subset of articles to be used for the attack pipeline
    def create_queries(set):
        #first load the corpus.jsonl file

        # then creat questinons using llms to

        # safe them to a queries.jsonl file
        pass


# --------------------------------------------------------------------------- #
# Example usage                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    CORPUS_PATH = "./pan-wikipedia-vandalism-corpus-2010"
    START_DIR = "./start"

    corpus = PANWVCCorpus(CORPUS_PATH, seed=42)

    # 1. Build the start subset
    written = corpus.build_start_subset(
        output_dir=START_DIR,
        subset_size=200,
    )
    print(f"Wrote {len(written)} start documents to {START_DIR}/")

    # 2. Apply a clean change
    clean = corpus.apply_change(START_DIR, change_type="clean")
    print(f"[CLEAN]      {clean.filename}  (editid={clean.editid})")

    # 3. Apply a vandalism change
    poison = corpus.apply_change(START_DIR, change_type="vandalism")
    print(f"[VANDALISM]  {poison.filename}  (editid={poison.editid})")
    
    