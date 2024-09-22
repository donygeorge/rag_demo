from langfuse import Langfuse
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader("data").load_data()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a retriever to fetch relevant documents
retriever = index.as_retriever(retrieval_mode='similarity', k=3)

# Create a query engine
# query_engine = index.as_query_engine()

langfuse = Langfuse()

# we use a very simple eval here, you can use any eval library
# see https://langfuse.com/docs/scores/model-based-evals for details
def llm_evaluation(output, expected_output):
    client = openai.OpenAI()
    
    prompt = f"""
    Compare the following output with the expected output and evaluate its accuracy:
    
    Output: {output}
    Expected Output: {expected_output}
    
    Provide a score (0 for incorrect, 1 for correct) and a brief reason for your evaluation.
    Return your response in the following JSON format:
    {{
        "score": 0 or 1,
        "reason": "Your explanation here"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the accuracy of responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    evaluation = response.choices[0].message.content
    try:
        result = eval(evaluation)  # Convert the JSON string to a Python dictionary
    except Exception as e:
        print(f"Error evaluating response: {e}")
        result = {"score": 0, "reason": "Error evaluating response"}
    
    # Debug printout
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    print(f"Evaluation Result: {result}")
    
    return result["score"], result["reason"]

from datetime import datetime


def custom_query(input):
    # Retrieve relevant documents
    relevant_docs = retriever.retrieve(input)
    
    print(f"Number of relevant documents: {len(relevant_docs)}")
    print("\n" + "="*50 + "\n")
    
    relevant_docs_str = ""
    for i, doc in enumerate(relevant_docs):
        relevant_docs_str += f"Document {i+1}:\n"
        relevant_docs_str += f"Document content: {doc.node.get_content()}...\n"
        relevant_docs_str += "\n" + "="*50 + "\n"
        
    print("Relevant documents: ", relevant_docs_str)
    
    
    # for i, doc in enumerate(relevant_docs):
    #     print(f"Document {i+1}:")
    #     print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
    #     print(f"Metadata: {doc.node.metadata}")
    #     print(f"Score: {doc.score}")
    #     print("\n" + "="*50 + "\n")

    # Create a prompt for the LLM
    prompt = f"""
    You are an AI assistant tasked with answering questions based on the following documents:
    {relevant_docs_str}
    
    Question: {input}
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with answering questions based on information provided in the documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content


 
def rag_query(input):
  
  generationStartTime = datetime.now()

  # response = query_engine.query(input)
  # output = response.response
  output = custom_query(input)
  print("input: ", input)
  print("output: ", output)
 
  langfuse_generation = langfuse.generation(
    name="strategic-plan-qa",
    input=input,
    output=output,
    model="gpt-3.5-turbo",
    start_time=generationStartTime,
    end_time=datetime.now()
  )
 
  return output, langfuse_generation

def run_experiment(experiment_name):
  dataset = langfuse.get_dataset("strategic_plan_qa_pairs")
 
  for item in dataset.items:
    completion, langfuse_generation = rag_query(item.input)
 
    item.link(langfuse_generation, experiment_name) # pass the observation/generation object or the id
 
    score, reason = llm_evaluation(completion, item.expected_output)
    langfuse_generation.score(
      name="accuracy",
      value=score,
      comment=reason
    )

run_experiment("Experiment 3")