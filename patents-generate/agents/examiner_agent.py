# agents/examiner_agent.py

# 假设 PatentDB 稍后会定义，这里先做一个前向声明或简单模拟
# from database.patent_db import PatentDB

class ExaminerAgent:
    def __init__(self, patent_db_handler=None):
        """
        审查智能体 (Examiner Agent)
        负责审查专利草稿，并利用 RRAG (检索增强生成) 机制提供反馈。
        """
        self.patent_db = patent_db_handler  # 依赖注入 PatentDB 的处理器
        print(f"ExaminerAgent initialized. PatentDB handler: {'Provided' if patent_db_handler else 'Not Provided'}")

    def _identify_key_claims_or_statements(self, pg_tree_node: dict) -> list:
        """
        (模拟) 从 PGTree 节点的生成内容中识别需要 RRAG 验证的关键声明或技术点。
        """
        node_title = pg_tree_node.get("title", "Unknown Node")
        content = pg_tree_node.get("generated_content", "")
        if not content:
            return []

        print(f"ExaminerAgent: Identifying key statements in '{node_title}' for RRAG.")
        # 实际应用中，这里可能使用 NLP 技术 (如实体识别、关系抽取) 或 LLM 来提取关键信息。
        # 简化模拟：假设我们提取内容中的某些关键词或短语作为检索查询。
        # 例如，如果内容包含“一种新颖的算法”，则可能将其作为检索点。

        # 模拟提取到的需要检索的查询点
        key_statements_for_retrieval = []
        if "权利要求" in node_title or "claim" in node_title.lower():  # 对权利要求部分特别关注
            # 假设权利要求内容中每句话都是一个关键点
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            key_statements_for_retrieval.extend(sentences[:2])  # 取前两个句子作为模拟
        elif "技术方案" in node_title:
            key_statements_for_retrieval.append(f"现有技术中关于'{content[:30]}...'的方案")

        if key_statements_for_retrieval:
            print(f"ExaminerAgent: Identified for RRAG in '{node_title}': {key_statements_for_retrieval}")
        return key_statements_for_retrieval

    def _perform_retrieval(self, queries: list) -> dict:
        """
        (模拟) 执行检索操作。
        实际应用中，会调用 patent_db 的高级搜索功能。
        """
        if not self.patent_db:
            print("ExaminerAgent: PatentDB handler not available. Skipping retrieval.")
            return {"queries_ran": queries, "results": "No DB access, retrieval skipped."}

        retrieval_results = {}
        print(f"ExaminerAgent: Performing retrieval for queries: {queries}")
        for query in queries:
            # 模拟调用 patent_db 的搜索方法
            # search_results = self.patent_db.search_patents(query, top_k=3)
            # 模拟返回结果
            search_results = [
                {"id": "patent_xyz", "title": "相关的现有技术专利XYZ", "similarity": 0.85,
                 "snippet": "一种类似的现有技术..."},
                {"id": "paper_abc", "title": "相关的技术论文ABC", "similarity": 0.79, "snippet": "研究表明该领域..."},
            ]
            retrieval_results[query] = search_results
            print(f"ExaminerAgent: Retrieval for query '{query}' returned {len(search_results)} results (simulated).")
        return {"queries_ran": queries, "results": retrieval_results}

    def _generate_feedback_with_rag(self, pg_tree_node: dict, retrieval_info: dict) -> str:
        """
        (模拟) 结合检索到的信息 (RAG) 生成审查反馈。
        """
        node_title = pg_tree_node.get("title", "Unknown Node")
        original_content = pg_tree_node.get("generated_content", "")

        print(f"ExaminerAgent: Generating feedback for '{node_title}' using RAG.")
        feedback = f"对节点 '{node_title}' 的审查意见：\n"
        feedback += f"原始内容片段：'{original_content[:100]}...'\n"

        if retrieval_info and retrieval_info.get("results") != "No DB access, retrieval skipped.":
            feedback += "检索到的相关信息摘要：\n"
            for query, results in retrieval_info["results"].items():
                feedback += f"  针对查询 '{query}':\n"
                if results:
                    for res_item in results:
                        feedback += f"    - {res_item['title']} (相似度: {res_item['similarity']:.2f}): {res_item['snippet'][:50]}...\n"
                else:
                    feedback += "    - 未找到强相关信息。\n"
            feedback += "\n建议：请参考上述检索信息，确认技术方案的新颖性和创造性。可能需要调整措辞或补充差异化特征。\n"
        else:
            feedback += "由于未能执行检索，无法提供基于现有技术的具体比对意见。请常规检查内容的清晰度和完整性。\n"

        # 模拟LLM基于原始内容和检索结果生成更具体的反馈
        # llm_refined_feedback = llm_call(prompt=f"Original content: {original_content}\nRetrieved_info: {retrieval_info}\nReview and provide feedback.")

        # 简单规则：如果检索到信息，标记为需要修改
        if retrieval_info and retrieval_info.get("results") and any(retrieval_info["results"].values()):
            pg_tree_node["status"] = "needs_revision"
            feedback += "\n状态更新：此节点标记为 'needs_revision'。\n"
        else:
            pg_tree_node["status"] = "approved_by_examiner"  # 简化：无检索或无结果则批准
            feedback += "\n状态更新：此节点标记为 'approved_by_examiner' (简化审批)。\n"

        print(
            f"ExaminerAgent: Feedback for '{node_title}': {feedback.splitlines()[0]}... Status: {pg_tree_node['status']}")
        return feedback

    def review_pg_tree_node(self, pg_tree_node: dict, full_pg_tree: dict):
        """
        审查 PGTree 中的单个节点。
        """
        node_title = pg_tree_node.get("title", "Unknown Node")
        node_status = pg_tree_node.get("status", "unknown")

        if node_status not in ["completed", "needs_review"]:  # 只审查已完成或明确标记为待审查的
            # print(f"ExaminerAgent: Skipping review for node '{node_title}' with status '{node_status}'.")
            return

        print(f"\nExaminerAgent: Reviewing node '{node_title}' (Status: {node_status}).")
        pg_tree_node["status"] = "under_examination"

        # 1. 识别关键声明以进行 RRAG
        key_statements = self._identify_key_claims_or_statements(pg_tree_node)

        retrieval_info = None
        if key_statements:
            # 2. 执行检索 (RRAG 的 R部分)
            retrieval_info = self._perform_retrieval(key_statements)
        else:
            print(
                f"ExaminerAgent: No key statements identified for RRAG in '{node_title}'. Proceeding with general review.")

        # 3. 生成反馈 (RRAG 的 G部分，结合检索结果)
        feedback = self._generate_feedback_with_rag(pg_tree_node, retrieval_info)

        # 存储反馈到节点中 (可以定义一个新字段，如 'examination_feedback')
        pg_tree_node["examination_feedback"] = feedback

        # print(f"ExaminerAgent: Node '{node_title}' review complete. New status: {pg_tree_node['status']}")

    def review_entire_patent_draft(self, pg_tree_root: dict):
        """
        递归审查整个 PGTree 草稿。
        """
        # 审查当前节点 (如果其内容已生成)
        if pg_tree_root.get("generated_content") and pg_tree_root.get("status") == "completed":
            self.review_pg_tree_node(pg_tree_root, pg_tree_root)  # 简化：full_pg_tree 传入自身

        # 递归审查子节点
        for child_node in pg_tree_root.get("children", []):
            self.review_entire_patent_draft(child_node)

        return pg_tree_root


# 示例用法 (用于测试)
if __name__ == '__main__':
    # 模拟 PatentDB 处理器
    class MockPatentDB:
        def search_patents(self, query, top_k=3):
            print(f"MockPatentDB: Searching for '{query}' (top_k={top_k}).")
            return [{"id": f"mock_id_{i}", "title": f"Mock Patent for {query} #{i + 1}", "similarity": 0.8 - i * 0.1,
                     "snippet": "This is a mock snippet."} for i in range(top_k)]


    # 先创建 Planner 和 Writer 来生成草稿
    from planner_agent import PlannerAgent
    from writer_agent import WriterAgent

    planner = PlannerAgent()
    writer = WriterAgent()
    idea = "一种用于检测水质污染的便携式光学传感器"
    plan = planner.generate_patent_plan(idea)
    draft = writer.populate_pg_tree(plan)

    print("\n--- Draft PGTree (before examination) ---")
    # import json
    # print(json.dumps(draft, indent=2, ensure_ascii=False))

    # 创建 ExaminerAgent 并审查草稿
    mock_db = MockPatentDB()
    examiner = ExaminerAgent(patent_db_handler=mock_db)
    examined_draft = examiner.review_entire_patent_draft(draft)

    print("\n--- Examined PGTree (after examination) ---")
    import json


    # print(json.dumps(examined_draft, indent=2, ensure_ascii=False))

    def print_pg_tree_examination_status(node, indent_level=0):
        print("  " * indent_level + f"- {node['title']} (ID: {node['id']}, Status: {node['status']})")
        if node.get('examination_feedback'):
            feedback_summary = node['examination_feedback'].splitlines()
            print("  " * (indent_level + 1) + f"  Feedback: {feedback_summary[0]}...")
        for child in node.get('children', []):
            print_pg_tree_examination_status(child, indent_level + 1)


    print("\nPGTree Structure with Examination Status:")
    print_pg_tree_examination_status(examined_draft)