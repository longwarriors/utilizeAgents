import asyncio
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from browser_use import Agent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_api_key():
    """加载API密钥"""
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY', '')
    if not api_key:
        raise ValueError('DEEPSEEK_API_KEY is not set')
    return api_key


def serialize_result(result):
    """将结果转换为可JSON序列化的格式"""
    if isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [serialize_result(item) for item in result]
    elif isinstance(result, (str, int, float, bool, type(None))):
        return result
    else:
        # 将不可序列化的对象转换为字符串表示
        return str(result)


def save_results(data, filename_prefix='result'):
    """保存结果到JSON文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 创建saved文件夹（如果不存在）
    save_dir = 'saved'
    os.makedirs(save_dir, exist_ok=True)
    # 使用os.path.join构建文件路径
    filename = os.path.join(save_dir, f'{filename_prefix}_{timestamp}.json')
    try:
        # 序列化结果，确保只包含可JSON化的数据
        serialized_data = serialize_result(data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=2)
        logger.info(f'Results saved to {filename}')
    except Exception as e:
        logger.error(f'Failed to save results: {str(e)}')
        raise


async def run_agent(task, filename_prefix='result'):
    """运行浏览器代理任务"""
    try:
        api_key = load_api_key()

        # 初始化Agent
        agent = Agent(
            task=task,
            llm=ChatOpenAI(
                base_url='https://api.deepseek.com/v1',
                model='deepseek-chat',  # 'deepseek-reasoner'
                api_key=SecretStr(api_key),
            ),
            use_vision=False,
        )

        # 运行任务并获取结果
        logger.info(f'Starting task: {task[:100]}...')
        result = await agent.run()

        # 保存结果
        save_results(result, filename_prefix)
        logger.info('Task completed successfully')
        return result

    except Exception as e:
        logger.error(f'Task failed: {str(e)}')
        raise


# 示例任务
TASKS = {
    'blog_search': """
        1. Go to https://www.google.com/
        2. Search for "苏剑林的科学空间 attention"
        3. Click on the first result that leads to 苏剑林's blog (likely kexue.fm)
        4. Find and click on the first blog post containing "attention" in the title or summary
        5. Return the title and URL of the page
    """,

    'news_collection': """
        1. Go to https://news.google.com/
        2. Search for "artificial intelligence"
        3. Collect titles and URLs of the top 3 news articles
        4. Return the collected information as a list
    """,

    'product_price': """
        1. Go to https://www.amazon.com/
        2. Search for "wireless mouse"
        3. Click on the first product
        4. Get the product title and price
        5. Return the product information
    """,

    'weather_check': """
        1. Go to https://www.weather.com/
        2. Search for "Beijing weather"
        3. Get the current temperature and weather condition
        4. Return the weather information
    """
}


async def main():
    """主函数，运行所有示例任务"""
    for task_name, task_description in TASKS.items():
        logger.info(f'Running {task_name} task...')
        result = await run_agent(task_description, task_name)
        # 序列化结果以确保打印时无错误
        serialized_result = serialize_result(result)
        print(f'Result for {task_name}:')
        print(json.dumps(serialized_result, ensure_ascii=False, indent=2))
        print('-' * 50)


if __name__ == '__main__':
    asyncio.run(main())
