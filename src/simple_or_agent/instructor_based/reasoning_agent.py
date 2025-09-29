from enum import Enum
import os
from typing import List, Optional

import instructor
from instructor import Mode
from pydantic import BaseModel, Field


class NextAction(str, Enum):
    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"
    RESET = "reset"


api_key = os.getenv("OPENROUTER_API_KEY")

class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description="A concise title summarizing the step's purpose")
    action: Optional[str] = Field(
        None, description="The action derived from this step. Talk in first person like I will ..."
    )
    result: Optional[str] = Field(
        None, description="The result of executing the action. Talk in first person like I did this and got ... "
    )
    reasoning: Optional[str] = Field(None, description="The thought process and considerations behind this step")
    next_action: Optional[NextAction] = Field(
        None,
        description="Indicates whether to continue reasoning, validate the provided result, or confirm that the result is the final answer",
    )
    confidence: Optional[float] = Field(None, description="Confidence score for this step (0.0 to 1.0)")


class ReasoningSteps(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description="A list of reasoning steps")

def get_system_prompt(min_steps: int = 1, max_steps: int = 10):
    return f"""\
    You are a meticulous, thoughtful, and logical Reasoning Agent who solves complex problems through clear, structured, step-by-step analysis.\n

    Step 1 - Problem Analysis:
        - Restate the user's task clearly in your own words to ensure full comprehension.
        - Identify explicitly what information is required and what tools or resources might be necessary.

        Step 2 - Decompose and Strategize:
        - Break down the problem into clearly defined subtasks.
        - Develop at least two distinct strategies or approaches to solving the problem to ensure thoroughness.

        Step 3 - Intent Clarification and Planning:
        - Clearly articulate the user's intent behind their request.
        - Select the most suitable strategy from Step 2, clearly justifying your choice based on alignment with the user's intent and task constraints.
        - Formulate a detailed step-by-step action plan outlining the sequence of actions needed to solve the problem.

        Step 4 - Execute the Action Plan:
        For each planned step, document:
        1. **Title**: Concise title summarizing the step.
        2. **Action**: Explicitly state your next action in the first person ('I will...').
        3. **Result**: Execute your action using necessary tools and provide a concise summary of the outcome.
        4. **Reasoning**: Clearly explain your rationale, covering:
            - Necessity: Why this action is required.
            - Considerations: Highlight key considerations, potential challenges, and mitigation strategies.
            - Progression: How this step logically follows from or builds upon previous actions.
            - Assumptions: Explicitly state any assumptions made and justify their validity.
        5. **Next Action**: Clearly select your next step from:
            - **continue**: If further steps are needed.
            - **validate**: When you reach a potential answer, signaling it's ready for validation.
            - **final_answer**: Only if you have confidently validated the solution.
            - **reset**: Immediately restart analysis if a critical error or incorrect result is identified.
        6. **Confidence Score**: Provide a numeric confidence score (0.0–1.0) indicating your certainty in the step's correctness and its outcome.

        Step 5 - Validation (mandatory before finalizing an answer):
        - Explicitly validate your solution by:
            - Cross-verifying with alternative approaches (developed in Step 2).
            - Using additional available tools or methods to independently confirm accuracy.
        - Clearly document validation results and reasoning behind the validation method chosen.
        - If validation fails or discrepancies arise, explicitly identify errors, reset your analysis, and revise your plan accordingly.

        Step 6 - Provide the Final Answer:
        - Once thoroughly validated and confident, deliver your solution clearly and succinctly.
        - Restate briefly how your answer addresses the user's original intent and resolves the stated task.

        General Operational Guidelines:
        - Ensure your analysis remains:
            - **Complete**: Address all elements of the task.
            - **Comprehensive**: Explore diverse perspectives and anticipate potential outcomes.
            - **Logical**: Maintain coherence between all steps.
            - **Actionable**: Present clearly implementable steps and actions.
            - **Insightful**: Offer innovative and unique perspectives where applicable.
        - Always explicitly handle errors and mistakes by resetting or revising steps immediately.
        - Adhere strictly to a minimum of {min_steps} and maximum of {max_steps} steps to ensure effective task resolution.
        - Execute necessary tools proactively and without hesitation, clearly documenting tool usage.
        - Only create a single instance of ReasoningSteps for your response.\
    """

def run_reasoning_agent(prompt: str):
    client = instructor.from_provider("openrouter/openai/gpt-oss-20b", api_key=api_key, mode=Mode.TOOLS)

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
            model="qwen/qwen3-next-80b-a3b-instruct",
            messages=messages,
            response_model=ReasoningSteps,
            extra_body={"provider": {"require_parameters": True}}
        )

    return response

def create_transcript(reasoning_steps: List[ReasoningStep]) -> str:
    transcript = ""
    for index, step in enumerate(reasoning_steps):
        transcript += f"Step {index + 1}: {step.title}\n"
        transcript += f"Action: {step.action}\n"
        transcript += f"Result: {step.result}\n"
        transcript += f"Reasoning: {step.reasoning}\n"
        transcript += f"Next Action: {step.next_action}\n"
        transcript += f"Confidence: {step.confidence}\n"
    return transcript


if __name__ == "__main__":
    response = run_reasoning_agent("Given this table defining * on the set S = {a, b, c, d, e}\n\n|*|a|b|c|d|e|\n|---|---|---|---|---|---|\n|a|a|b|c|b|d|\n|b|b|c|a|e|c|\n|c|c|a|b|b|a|\n|d|b|e|b|e|d|\n|e|d|b|a|d|c|\n\nprovide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order.")
    print("Reasoning Steps: ", response.reasoning_steps)
    print("\n")
    print("Final Answer: ", response.reasoning_steps[-1].result)
    transcript = create_transcript(response.reasoning_steps)
    print("Transcript: ", transcript)