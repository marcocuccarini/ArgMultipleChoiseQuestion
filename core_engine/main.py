# main.py

import json
from classes.LLM import LLM
from classes.LLMUser import LLMUser
from classes.AF import ArgumentationGraph

def main():
    # 1️⃣ Initialize the low-level LLM
    llm_model = LLM(model="gemma3:4b")

    # 2️⃣ Create the high-level LLM user
    user = LLMUser(llm_model)

    # 3️⃣ Text to analyze
    text = (
        "Governments should invest more in renewable energy sources such as solar, wind, and hydroelectric power. "
        "Renewable energy reduces greenhouse gas emissions, mitigates climate change, and creates long-term economic opportunities through new green jobs. "
        "Furthermore, investing in renewables decreases dependence on fossil fuels, improving national energy security. "
        "However, critics argue that renewable energy projects are expensive to implement, require significant land use, and sometimes disrupt local communities. "
        "For example, large solar farms can displace wildlife habitats, and wind turbines may affect bird migration patterns. "
        "There are also concerns about the intermittency of renewable energy, as solar and wind output depends on weather conditions, which can affect reliability. "
        "Supporters respond that technological advancements and improved storage solutions, such as batteries, are addressing these intermittency issues. "
        "They also argue that the long-term economic and environmental benefits outweigh the initial costs, and that proper planning can minimize community and ecological impacts. "
        "Some even point out that countries investing in renewables are positioning themselves as leaders in future global energy markets, gaining both economic and geopolitical advantages. "
        "On the other hand, some policymakers emphasize the importance of a balanced energy strategy that includes gradual integration of renewables alongside cleaner fossil fuel technologies to ensure a stable energy supply while transitioning to a low-carbon future. "
        "Debates continue about how to achieve the optimal balance between environmental sustainability, economic feasibility, and social acceptance."
    )

    # 4️⃣ Initialize the ArgumentationGraph
    graph_builder = ArgumentationGraph()

    # 5️⃣ Build the graph from text
    result = graph_builder.build_from_text(text, user)

    # 6️⃣ Print results
    print("\n=== Extracted Arguments ===")
    print(json.dumps(result["arguments"], indent=2))

    print("\n=== Detected Relations ===")
    print(json.dumps(result["relations"], indent=2))

    print("\n=== Graph Nodes & Strengths ===")
    print(json.dumps(result["graph"]["nodes"], indent=2))

    print("\n=== Graph Edges ===")
    print(json.dumps(result["graph"]["edges"], indent=2))

    print("\n=== Computed Strengths ===")
    print(json.dumps(result["strengths"], indent=2))

if __name__ == "__main__":
    main()
