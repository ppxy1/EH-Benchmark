from __future__ import annotations
import warnings
from typing import List, Optional
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import BaseTool
from pydantic import Field, field_validator
import os
import time
import traceback
import random
import json
from urllib.parse import urljoin, urlparse
from collections import deque
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "your-api-key-here")

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")

class RAGTool(BaseTool):
    name: str = "RAGTool"
    description: str = (
        "Retrieve relevant medical context from predefined webpages and their linked pages "
        "to provide background information for user queries."
    )

    openai_api_key: str = Field(..., description="OpenAI API key")

    url_list: List[str] = Field(
        default_factory=lambda: [
            "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases",
            "https://www.health.tas.gov.au/health-topics/eyes-and-vision-ophthalmology/common-eye-conditions",
            "https://my.clevelandclinic.org/health/diseases/open-angle-glaucoma",
            "https://my.clevelandclinic.org/health/diseases/angle-closure-glaucoma",
        ]
    )

    custom_qa_prompt: PromptTemplate = Field(
        default_factory=lambda: PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a trusted medical Retrieval-Augmented-Generation expert.\n\n"
                "Context:\n{context}\n\nQuestion:\n{question}\n\n"
                "Instructions:\n"
                "• Use only the context to determine if the answer to the multiple-choice question can be found.\n"
                "• If nothing is relevant, answer exactly: No relevant data found.\n\n"
                "Answer:"
            ),
        )
    )

    @field_validator("openai_api_key")
    def _check_key(cls, v):
        if not v:
            raise ValueError(
                "openai_api_key is required. "
                "Pass it explicitly or set env OPENAI_API_KEY."
            )
        return v

    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    ]

    def _get_headers(self):
        return {
            "User-Agent": random.choice(self._USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

    def _fetch_raw_text(self, url: str, retries: int = 3) -> Optional[str]:
        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=self._get_headers(), timeout=15)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    main_content = soup.find("main")
                    if main_content:
                        text = main_content.get_text(separator=" ", strip=True)
                    else:
                        for element in soup.find_all(["header", "footer", "nav", "aside"]):
                            element.decompose()
                        text = soup.get_text(separator=" ", strip=True)
                    return text
                time.sleep(0.5)
            except Exception as e:
                print(f"[RAGTool] Error fetching {url} after {attempt + 1} attempts: {e}")
                time.sleep(0.5)
        return None

    def get_page_links(self, url: str) -> List[str]:
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=15)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=True)
            domain = urlparse(url).netloc
            relevant_links = set()
            for link in links:
                href = link["href"]
                absolute_url = urljoin(url, href)
                absolute_parsed = urlparse(absolute_url)
                if (absolute_parsed.netloc == domain and absolute_parsed.scheme in ("http", "https")):
                    root_url = absolute_url.split("#")[0]
                    relevant_links.add(root_url)
            return list(relevant_links)
        except Exception as e:
            print(f"[RAGTool] Error getting links from {url}: {e}")
            return []

    def _run(self, input: str) -> str:
        query = input
        all_docs = []
        fetched_urls = set()
        queue = deque([(url, 0) for url in self.url_list])
        MAX_URLS = 150
        MAX_DEPTH = 2

        while queue and len(fetched_urls) < MAX_URLS:
            url, depth = queue.popleft()
            root_url = url.split("#")[0]
            if root_url in fetched_urls or depth > MAX_DEPTH:
                continue
            fetched_urls.add(root_url)
            print(f"[RAGTool] Crawling {root_url} (depth: {depth})")
            raw_text = self._fetch_raw_text(url)
            if raw_text and len(raw_text.strip()) > 0:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
                docs = [Document(page_content=chunk, metadata={"source": url}) for chunk in splitter.split_text(raw_text)]
                all_docs.extend(docs)
            linked_urls = self.get_page_links(url)
            for linked_url in linked_urls:
                if linked_url not in fetched_urls:
                    queue.append((linked_url, depth + 1))

        if not all_docs:
            return json.dumps({"background_info": "No relevant data found."})

        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vectordb = Chroma.from_documents(all_docs, embeddings)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        retrieved = retriever.invoke(query)

        if not retrieved:
            return json.dumps({"background_info": "No relevant data found."})

        background_info = "\n\n".join([f"Information from {doc.metadata['source']}:\n{doc.page_content}" for doc in retrieved])
        return json.dumps({"background_info": background_info})

    async def _arun(self, input: str) -> str:
        return self._run(input)