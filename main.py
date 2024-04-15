from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.tools import PythonREPLTool


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

    #csv_agent.run("In the file episode_info.csv, Which writer wrote the most episodes? How many episodes did he write?")
    csv_agent.run("Print in ascending order the number of episodes in each season")

if __name__ == "__main__":
    main()
