# agents/writer_agent.py

class WriterAgent:
    def __init__(self):
        """
        编写智能体 (Writer Agent)
        负责根据 PGTree 结构撰写专利的具体内容。
        """
        print("WriterAgent initialized.")

    def _write_node_content(self, pg_tree_node: dict, full_pg_tree: dict) -> str:
        """
        (模拟)为 PGTree 中的单个节点生成内容。
        在实际应用中，这里会调用 LLM 或其他内容生成逻辑。

        Args:
            pg_tree_node (dict): 当前需要填充内容的 PGTree 节点。
            full_pg_tree (dict): 完整的 PGTree，可能用于获取上下文信息。

        Returns:
            str: 为该节点生成的内容。
        """
        node_id = pg_tree_node["id"]
        node_title = pg_tree_node["title"]
        guideline = pg_tree_node["content_guideline"]

        # 模拟内容生成
        print(f"WriterAgent: Writing content for node '{node_title}' (ID: {node_id}) based on guideline: '{guideline}'")

        # 实际应用中，这里会是复杂的逻辑，可能包括：
        # 1. 理解节点的上下文 (父节点、兄弟节点)。
        # 2. 调用 LLM，并将 guideline 和上下文作为提示的一部分。
        # 3. 处理 LLM 的输出。
        # 4. (如果适用) 从数据库或其他来源检索参考信息。

        # 模拟LLM返回的内容
        generated_text = f"[这是为 '{node_title}' (ID: {node_id}) 生成的模拟内容。遵循指南：'{guideline}。']"

        # 模拟token消耗的追踪 (如果TokenTracker集成在此处)
        # self.token_tracker.add_tokens("writer_agent_generation", len(generated_text))

        return generated_text

    def populate_pg_tree(self, pg_tree_root: dict) -> dict:
        """
        遍历 PGTree 并为每个需要填充的节点生成内容。
        这是一个递归函数。

        Args:
            pg_tree_root (dict): PGTree 的根节点或当前处理的子树的根节点。

        Returns:
            dict: 更新了 generated_content 和 status 的 PGTree。
        """
        if pg_tree_root["status"] == "pending":  # 只处理待处理的节点
            # 检查是否有 content_guideline，表明这是一个需要填充内容的叶节点或父节点本身也需要内容
            if pg_tree_root.get("content_guideline"):
                pg_tree_root["status"] = "in_progress"
                # print(f"WriterAgent: Processing node '{pg_tree_root['title']}' for content generation.")

                # 调用内部方法生成内容
                generated_content = self._write_node_content(pg_tree_root, pg_tree_root)  # 简化：这里 full_pg_tree 传入自身
                pg_tree_root["generated_content"] = generated_content
                pg_tree_root["status"] = "completed"  # 假设一次性完成，实际可能需要更复杂的状态管理
                print(f"WriterAgent: Node '{pg_tree_root['title']}' content generated and status set to 'completed'.")

        # 递归处理子节点
        for child_node in pg_tree_root.get("children", []):
            self.populate_pg_tree(child_node)  # 递归调用

        return pg_tree_root


# 示例用法 (用于测试)
if __name__ == '__main__':
    # 先创建一个 PlannerAgent 来生成计划
    from planner_agent import PlannerAgent

    planner = PlannerAgent()
    idea = "一种可折叠的便携式太阳能充电器"
    patent_plan_tree = planner.generate_patent_plan(idea)

    print("\n--- Initial PGTree (before writing) ---")
    import json

    # print(json.dumps(patent_plan_tree, indent=2, ensure_ascii=False))

    # 创建 WriterAgent 并填充 PGTree
    writer = WriterAgent()
    populated_tree = writer.populate_pg_tree(patent_plan_tree)

    print("\n--- Populated PGTree (after writing) ---")


    # print(json.dumps(populated_tree, indent=2, ensure_ascii=False))

    def print_pg_tree_content_status(node, indent_level=0):
        print("  " * indent_level + f"- {node['title']} (ID: {node['id']}, Status: {node['status']})")
        if node['generated_content']:
            print("  " * (indent_level + 1) + f"  Content: {node['generated_content'][:100]}...")  # 打印部分内容
        for child in node.get('children', []):
            print_pg_tree_content_status(child, indent_level + 1)


    print("\nPGTree Structure with Content Status:")
    print_pg_tree_content_status(populated_tree)