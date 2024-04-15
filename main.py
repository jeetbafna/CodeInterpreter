from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.tools import PythonREPLTool

# from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.initialize import initialize_agent
from langchain_core.tools import Tool

load_dotenv()


def main():
    print("Start")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # python_agent_executor.run(
    #     "Generate and save to current working directory 5 QR Codes that point to www.github.com/jeetbafna, you have qrcode package installed already"
    # )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        path="./episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # csv_agent.run("In the file episode_info.csv, Which writer wrote the most episodes? How many episodes did he write?")
    #csv_agent.run("Print in ascending order the number of episodes in each season")

    tools_for_agent = [
        Tool(
            name="PythonAgent",
            func=python_agent_executor.run,
            description="""useful when you need to transform natural language and write from it python and execute the python code
            returning the results of the code execution,
            DO NOT SENF PYTHON CODE TO THIS TOOL""",
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent.run,
            description="""useful when you need to answer questions over episode_info.csv file,
            takes an input the entire question and returns the answer after running pandas calculation""",
        ),
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    grand_agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )
    # grand_agent.run(
    #     "Generate and save to current working directory 5 QR Codes that point to www.github.com/jeetbafna, you have qrcode package installed already"
    # )

    grand_agent.run("Using the file episode_info.csv, Print in ascending order the number of episodes in each season")

if __name__ == "__main__":
    main()
