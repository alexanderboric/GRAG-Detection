import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path("/Users/alexanderboric/Desktop/Uni/Bachelor/GraphPoison/.env"))
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY","").strip(), base_url=os.getenv("OPENAI_BASE_URL","").strip())
print("base_url:", client.base_url)
try:
    resp = client.chat.completions.create(model="mistral-small-4", messages=[{"role":"user","content":"say hi in one word"}])
    print(resp.choices[0].message.content)
except Exception as e:
    import traceback; traceback.print_exc()
