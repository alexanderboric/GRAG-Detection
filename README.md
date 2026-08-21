# GraphPoison
Analysing poisoning attacks on graph based RAG Systems

This Repo serves multiple purposes: 
1. Reproduce and analyse the results of an attack on graph-based RAG systems like TKPA, GragPoison or logicPoison https://github.com/Jord8061/logicPoison on smaller local datasets and models. (e.g. LLaMA 3.1-8B-Instruct,etc. )
2. Developing detection methods for poisoning attacks on graph-based RAG systems.
3. Framework to evaluate Detection of Poisoning Attacks on Graph-Based RAG Systems.

# Structure of the Repo: 

The repo consits of the following folder: 
- data: This folder contains the datasets used for training and evaluation.
- models: This folder contains the pre-trained models and any custom models used in the project.
- attacks: This folder contains the implementation of the poisoning attacks on graph-based RAG systems. i.E 

## Installation

Installing Microsoft GraphRag
installing the requirements for the project
pip3 install -r requirements.txt
python -m spacy download en_core_web_trf

# Datasets: 

HotpotQA/2WikiMultihopQA/MuSiQue
https://huggingface.co/datasets/Jord8061/datasets?clone=true

Usage: #






## List of potential attacks to implement:
- PoisonedRag: https://github.com/sleeepeer/PoisonedRAG
- TKPA: https://arxiv.org/abs/2211.13227
- GragPoison: https://arxiv.org/abs/2211.13227
- logicPoison:
- KEPo: https://arxiv.org/abs/2603.11501
- ADMIT: https://arxiv.org/abs/2510.13842

Idea: 
- Use Leiden Clustering to identify clusters
- On upload, identify the cluster affected by upload
- Querry a benchmark to identify if specific requests are overshadowed by the attack

Problems: 

- Clean benchmark ingestions are always additions not changes
- Without examples for clean changes my detection method would be biased towards identifying changes as attacks, which is not the case in the real world, where most changes are clean.
- (Possible Solution) Wikipedia Documents + Edits, which haven't been reverted

## Questions

- Wie lang ist die Präsentation  10-15. max 20. Minuten 
- Welches Format ( Folien, Demo, etc.) Kein Vorwissen, viel Einleiten, Ziel und Pläne gut darstellen. Folien sinnvoll 
- Absprache der Wikipedia idee 
- Wann folgen weitere Vorträge anderer Personen ? 
- Zuerst strukturen , Parallel zum Doing schon schreiben 
- Früh mit schreiben anfagen, Zeitenschätzung besser und schnelleres Feedback 

Orientieren, und recherchieren ob es ähnliche Ansätze gibt, 
Anfangen zu schreiben 
Oberseminar ohne Termine und keine Festen Zeiten 
Nächste Woche kurz nach 12.
17. Juni erst um 13 Uhr. 


Präsentation: 

Satzweise entfernen stadt ganzes Document und danach als clean change einfügen. 

gerne weniger, auch nur 20min für endpräsentation
langsamer reden 
selektieren 
mehr abbildungen ! 
Direkt mit Graphrag einsteigen statt mit Vektore evtl. vektor nur in der lokalen suche nutzen 
Seitenzahlen bei den Folien 
Angreifermodel früher nennen (maybe weniger detailiert)
Ziel soll klar werden !!!! !!! Am besten direkt vor der Methodik 
Sonst gut 

Ausarbeitung: 

Titel anpassen zu lang als title aber als Forschungsfrage gut 
Zum Ende nen überblick über verfahren und das was noch geschehen wird und was einen erwartet und haupt ergenisse und Erkenntnisse nennen 
Abstract kurz und nenn auch schon Ergebnisse 

Ruhig introduction ein wenig kürzen, 

Kaptitel 2. Detailierte Einführung in den Hintergrund also nochmal genau erklärung z.B. was rag ist und wie es genau funktioniert

Abstract is optional aus seiner sicht, weiß nicht, ob das verpflichtend ist
Vielleicht sogar abstract auf deutsch notwenig ( Evlt sogar. im Dokument ) Prüfungsamt anfragen 
Stil ist gut, wissenschaftlich bleiben auch gut 

Detection needs a Change : Detection on changes instead of insertions

Testing different models 
Testing different document lengths, ( checking where the changes lie, adding e.g. 4-5 sentences padding around it and then cutting the it to lenght by removing trailing sentences)
Evaluate over poisoned Wikipedia


