# Retrieval augmented generation consists of two initial steps: indexing and retrieval.

### Indexing
# indexing describes the process of transforming an external (usually text) data source into
# a format that can be efficiently queried. This is typically done by transforming the (e.g.) text
# into a vector space representation. The vectors are then stored in a database that allows for
# efficient similarity search. 

### Retrieval
# retrieval is the process of querying the index with a query vector and returning the most similar
# vectors from the index. The query vector can be a question asked by a user, a text prompt, or any other
# text input. The retrieved vectors can then be used to generate a response.

# The following script uses llama_index alongside Ollama to query a PDF document for relevant information.
# We will use the open access CC licensed paper Sarstedt et al. (2024) as the document to index,
# query and play with.
# You can find it here https://onlinelibrary.wiley.com/doi/full/10.1002/mar.21982 or in the data folder.
import os
from utils import get_index, get_summary
from llama_index.core import SimpleDirectoryReader, Settings, get_response_synthesizer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.agent import ReActAgent

Settings.llm = Ollama(model="gemma2:9b", request_timeout=60, temperature=0.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", ollama_additional_kwargs={"mirostat": 0})

# standard indexing and retrieval setup
pdf_path = os.path.join("data","sarstedt_2024.pdf")
sarstedt = SimpleDirectoryReader(input_files=[pdf_path])
pdf = sarstedt.load_data()
index = get_index(pdf, "sarstedt")
retriever = VectorIndexRetriever(index, similarity_top_k=5)
response_synth = get_response_synthesizer(llm=Settings.llm)

# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synth,
#     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
#     )
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    )
# engine = index.as_query_engine()

# summary setup
pdf_path = os.path.join("data","sarstedt_2024.pdf")
sarstedt = SimpleDirectoryReader(input_files=[pdf_path])
pdf = sarstedt.load_data()
summary = get_summary(pdf, "sarstedt")
summary_engine = summary.as_query_engine(response_mode="tree_summarize")

tools = [
    # QueryEngineTool(
    #     query_engine=query_engine,
    #     metadata=ToolMetadata(
    #         name="sarstedt",
    #         description="Contains the Sarstedt et al. (2024) paper",
    #     )
    # ),
    QueryEngineTool(
        query_engine=summary_engine,
        metadata=ToolMetadata(
            name="sarsted_summary",
            description="provides summaries for the sarstedt et al. 2024 paper",
        )
    )
]
context = """Purpose: The primary role of this agent is to provide acurate answers for the paper of Sarstedt et al. 2024.
"""
agent = ReActAgent.from_tools(tools=tools, llm=Settings.llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    response = agent.chat(prompt)
    print(response)
    
# provide me a list of 1. general 2. specifc and 3. application focused questions for this paper