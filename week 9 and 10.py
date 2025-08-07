def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

model_id = 'mistralai/mixtral-8x7b-instruct-v01'

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.2, # this randomness or creativity of the model's responses 
}

credentials = {
    "url": "..."
}

project_id = "..."

model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

llm = WatsonxLLM(model_id=model)
role = """
    Dungeon & Dragons game master
"""

tone = "engaging and immersive"

template = """
    You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
    
    Answer:
    
"""
prompt = PromptTemplate.from_template(template)

def format_prompt(variables):
    return prompt.format(**variables)

# Create the LCEL chain
roleplay_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

# Create an interactive chat loop
while True:
    query = input("Question: ")
    
    if query.lower() in ["quit", "exit", "bye"]:
        print("Answer: Goodbye!")
        break
        
    response = roleplay_chain.invoke({"role": role, "question": query, "tone": tone})
    print("Answer: ", response)