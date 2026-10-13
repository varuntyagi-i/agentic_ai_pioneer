from crewai import Crew, Agent, Task, LLM
from crewai_tools import DallETool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any
import requests
import os

# Set up API key

os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"

llm = LLM(model="gpt-4o-mini")

class DownloadImageInput(BaseModel):
    image_url: str = Field(..., description="The URL of the image to download")

class DownloadImageTool(BaseTool):
    name: str = "Image Downloader Tool"
    description: str = "Downloads image from url"
    args_schema: Type[BaseModel] = DownloadImageInput

    def _run(self, image_url: str) -> str:
        filename = f"{os.path.basename(image_url)}.png"

        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Failed {e}"

        with open(filename, "wb") as image_file:
            for chunk in response.iter_content(chunk_size=8192):
                image_file.write(chunk)

        return f"Save as {filename}"


dalle_tool = DallETool(model="dall-e-3",
                       size="1024x1024",
                       quality="standard",
                       n=1)

prompt_improver_agent = Agent(
    llm=llm,
    role="Prompt improver",
    backstory="Ai agent that enhances prompts to be more detailed",
    goal="Improve text prompts for optimal image generation",
    verbose=True
)

image_generator_agent = Agent(
    llm=llm,
    role="Image generator",
    backstory="An AI agent that generates images based on detailed prompts, using DALLE and save the image in the current directory",
    goal="Generate images from enhanced prompts using DALLE and save them locally",
    tools=[dalle_tool, DownloadImageTool()],
    verbose=True
)

enhance_prompt_task = Task(
    description="Improve this prompt ({initial_prompt}) to make it more descriptive and detailed for image generation.",
    expected_output='"image_description": "Enhanced Prompt here',
    agent=prompt_improver_agent
)

generate_image_task = Task(
    description="Generate an image from the enhanced prompt using DALLE",
    expected_output="image_url: URL of the generated image",
    agent=image_generator_agent
)

download_image_task = Task(
    description="Download the generated image from the provided URL",
    expected_output="Image downloaded and saved locally",
    agent=image_generator_agent
)

crew = Crew(
    agents=[prompt_improver_agent, image_generator_agent],
    tasks=[enhance_prompt_task, generate_image_task, download_image_task]
)

crew.kickoff(inputs={'initial_prompt': 'A futuristic cityscape at sunset'})