import os
import re
from datetime import datetime, timedelta

import feedparser
import markdown
import pytz
import requests
import tiktoken
import yaml
import zhipuai
from bs4 import BeautifulSoup
from dateutil import parser
from feedgen.feed import FeedGenerator
from openai import ChatCompletion
from requests.exceptions import RequestException
from dotenv import load_dotenv
import pickle
import time
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建一个日志记录器
logger = logging.getLogger('default')


def get_for_openai_key():
    """
    获取 openai key
    Returns:
        openai key
    """
    for i in range(0, 5):
        res = request_for_openai_key()
        if res is not None and res != "":
            return res
    return None


def request_for_openai_key():
    """
    请求 openai key
    Returns:
        openai key
    """
    url = 'https://v1.luoxin.vip/api/get_random_key'

    headers = {
        'Content-Type': 'application/json',
        'authorization': f"Bearer {REQ_TOKEN}"
    }

    try:
        response = requests.post(url, headers=headers)
    except Exception as e:
        logger.warning(f"请求发生异常 {str(e)}")
        return ""

    if response.status_code == 200:
        # 请求成功
        logger.info(f"openai_key 请求成功: {response.json()}")
        return response.json()['token']
    else:
        logger.warning(f"openai_key 请求失败: {response.text}")
        return None


def html_to_text(html):
    """
    从 HTML 中提取纯文本
    Args:
        html: HTML

    Returns:
        纯文本
    """
    # 使用Beautiful Soup解析HTML
    soup = BeautifulSoup(html, 'html.parser')

    # 使用prettify()方法来保留HTML排版
    pretty_text = soup.prettify()

    # 从HTML中提取纯文本
    text = BeautifulSoup(pretty_text, 'html.parser').get_text("\n")

    return text


def is_new(entry):
    """
    判断文章是否是新文章
    Args:
        entry: 文章 (feedParse 解析的结果)

    Returns:
        True or False
    """
    # 尝试解析发布时间
    published_time = entry.get('published', entry.get('updated', ''))
    if published_time == '':
        return False
    else:
        return is_new_by_date(parser.parse(published_time))


def is_new_by_date(published_time):
    """
    判断文章是否是新文章
    Args:
        published_time: 文章的发布时间

    Returns:
        True or False
    """
    # 获取当前时间，并添加时区信息（UTC）
    current_time = datetime.now(pytz.utc)

    # 计算前一天的日期
    one_day_ago = current_time - timedelta(days=3)

    if published_time.tzinfo is None:
        published_time = pytz.UTC.localize(published_time)
    # 如果文章的发布时间在前一天内，认为它是新文章
    return published_time >= one_day_ago


def add_entry_to_feed(rss_feed, entry, summarize: bool = False, order='prepend'):
    """
    将 entry 添加到 rss_feed 中
    Args:
        rss_feed: rss 生成器
        entry: 待添加的文章 (feedParse 解析的结果)
        summarize: 是否使用 AI 总结
        order: 添加的顺序

    Returns:

    """
    published_time_str = entry.get('published', entry.get('updated', None))
    if published_time_str is None:
        published_time = datetime.now(pytz.utc)
    else:
        published_time = parser.parse(published_time_str)

    # 添加文章到新Feed
    fe = rss_feed.add_entry(order=order)
    fe.title(entry.title)
    fe.link(href=entry.link)
    content = entry.get('content', [{}])[0].get('value', None)
    if content is None:
        content = entry.get('description', None)
    if summarize and content is not None:
        try:
            plain_text = html_to_text(content)
            summary = gpt_summary(plain_text)
            if summary is None:
                logger.info("文章较短，不需要总结")
            else:
                summary_context = markdown.markdown(summary['content'].replace("\n", "\n\n"), extensions=['sane_lists'])
                content = f"{summary_context} <hr/><br/> {content}"
                # 统计用的总 token 数
                feed_title = rss_feed.title()
                model = summary['model']
                total_tokens = summary['usage']['total_tokens']

                feed_token_cost.setdefault(feed_title, {})
                feed_token_cost[feed_title].setdefault(model, 0)
                feed_token_cost[feed_title][model] += total_tokens

                logger.info(f"Summarization cost {summary['usage']}")
        except Exception as e:
            logger.warning(f"Summarize failed. {str(e)}")

    fe.description(content)
    fe.author(name=getattr(entry, "author", {}))
    fe.pubDate(published_time)


def gpt_summary(text):
    """
    使用 AI 总结
    Args:
        text: 输入的文本

    Returns:
        None or {content, usage, model}
    """
    chinese_count = count_chinese_characters(text)
    if chinese_count > 300:
        return summarize_with_zhipuai(concat_zhipuai_prompt(text), ZHIPUAI_MODEL)
    else:
        words = text.split()  # 使用空格拆分文本为单词列表
        if len(words) < 300:
            return None
        tokens_num = calculate_tokens_with_openai(text, "cl100k_base")
        if tokens_num < 4000 - 100:
            model = "gpt-3.5-turbo-0613"
        else:
            model = "gpt-3.5-turbo-16k-0613"
            if tokens_num > 16000 - 100:
                # 每个token大概对应四个字母，来源：https://platform.openai.com/tokenizer
                text = text[:16000 * 3]
        return summarize_with_openai(concat_openai_prompt(text), model=model)


def concat_openai_prompt(text: str) -> list:
    return [
        {"role": "system",
         "content": f"您是一位全能专家，不仅了解各个领域的知识，而且擅长总结归纳。我会提供一篇文章，你需要从中提取出5个关键词，并用中文归纳出文章的大纲，最后在总结一个综合摘要。摘要应涵盖原始文本中呈现的所有关键点和主要观点，同时将信息压缩成简明易懂的格式。请确保摘要包含支持主要观点的相关细节和示例，同时避免任何不必要的信息或重复内容。摘要长度应适合原始文本的长度和复杂性，提供清晰准确的概述，并不遗漏任何重要信息。同时你的输出内容还需要满足以下要求：1.以严格的markdown格式输出， 2.“关键词”，“大纲”，“总结”这几个字需要加粗 3.上述三个部分的内容应该分段落"},
        {"role": "user", "content": text}
    ]


def concat_zhipuai_prompt(text: str) -> list:
    return [
        {"role": "user",
         "content": f"您是一位全能专家，不仅了解各个领域的知识，而且擅长总结归纳。我会提供一篇文章，你需要从中提取出5个关键词，并用中文归纳出文章的大纲，最后在总结一个综合摘要。摘要应涵盖原始文本中呈现的所有关键点和主要观点，同时将信息压缩成简明易懂的格式。请确保摘要包含支持主要观点的相关细节和示例，同时避免任何不必要的信息或重复内容。摘要长度应适合原始文本的长度和复杂性，提供清晰准确的概述，并不遗漏任何重要信息。同时你的输出内容还需要注意“关键词”，“大纲”，“总结”三个部分的内容应该分段落. 下面是文章的具体内容：\n\n{text}"}
    ]


def calculate_tokens_with_openai(text: str, encoding_name: str) -> int:
    """
    计算 text 的 token 数
    Args:
        text: 文本
        encoding_name: 编码名称

    Returns:
        token 数
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def count_chinese_characters(text):
    """
    统计中文字符的数量
    Args:
        text: 文本

    Returns:
        中文字符的数量
    """
    pattern = re.compile(r'[\u4e00-\u9fa5]')  # 匹配中文字符的正则表达式
    chinese_chars = re.findall(pattern, text)  # 使用正则表达式找到所有中文字符
    count = len(chinese_chars)  # 统计中文字符的数量
    return count


def summarize_with_zhipuai(prompt, model):
    """
    使用 zhipuai 总结
    Args:
        prompt: 输入的文本
        model: 使用的模型

    Returns:
        {"content": response["data"]["choices"][0]["content"], "usage": response['data']['usage'], "model": model}
    """
    response = zhipuai.model_api.invoke(
        model=model,
        temperature=0.3,
        return_type="text",
        prompt=prompt
    )
    return {"content": response["data"]["choices"][0]["content"], "usage": response['data']['usage'],
            "model": model}


def summarize_with_openai(prompt, model):
    """
    使用 openai 总结
    Args:
        prompt: 输入的文本
        model: 使用的模型

    Returns:
        {"content": response["data"]["choices"][0]["content"], "usage": response['data']['usage'], "model": model}
    """
    chat = ChatCompletion.create(
        api_base=OPENAI_API_BASE,
        model=model,
        api_key=OPENAI_API_KEY,
        messages=prompt,
    )
    return {"content": chat["choices"][0]["message"]["content"], "usage": dict(chat['usage']), "model": chat['model']}


def deep_copy(obj):
    """
    深拷贝对象
    Args:
        obj: 需要拷贝的对象

    Returns:
        拷贝后的对象
    """
    # 将对象序列化为二进制数据
    serialized_data = pickle.dumps(obj)
    # 反序列化二进制数据为对象
    loaded_data = pickle.loads(serialized_data)
    return loaded_data


def read_cache(filename: str) -> set:
    """
    读取缓存文件，返回一个 set
    Args:
        filename: 缓存文件名

    Returns:
        已生成的文章链接的 set
    """

    cache = set()
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            for line in file:
                link = line.strip()
                cache.add(link)
    return cache


def parse_url(con: {}, rss_feed: FeedGenerator, url: str) -> int:
    """
    解析 url，将新文章添加到 rss_feed 中
    Args:
        con: 配置
        rss_feed: rss 生成器
        url: 待解析的 url

    Returns: 新文章的数量

    """
    count = 0
    try:
        feed = feedparser.parse(url)
        if feed['bozo']:
            logger.info(f"解析 {url} 失败 {str(feed['bozo_exception'])}")
        for entry in feed.entries:
            if is_new(entry) \
                    and entry.link not in feed_url_set_cache[rss_feed.title()]:
                add_entry_to_feed(rss_feed, entry, con['summarize'])
                feed_url_set_cache[rss_feed.title()].add(entry.link)
                count += 1
            else:
                break
    except Exception as e:
        logger.warning(f"Failed to fetch URL: {url}. Error: {str(e)}")
    return count


def append_write_cache(cache_filename: str, link_to_append: set):
    """
    将 link_to_append 的内容 append 到 cache_filename
    Args:
        cache_filename: 缓存文件名
        link_to_append: 需要 append 的内容

    Returns: None

    """
    with open(cache_filename, 'a') as file:
        for link in link_to_append:
            file.write(link + '\n')


load_dotenv()

ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY", None)
ZHIPUAI_MODEL = os.environ.get('ZHIPUAI_MODEL', 'chatglm_std')
REQ_TOKEN = os.environ.get('REQ_TOKEN', None)


try:
    OPENAI_API_KEY = get_for_openai_key()
except Exception as e:
    logger.warning(f"获取 openai key 失败 {str(e)}")
    OPENAI_API_KEY = None
if OPENAI_API_KEY is None:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)


with open('config.yaml') as yaml_file:
    cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)['rss']
OPENAI_API_BASE = cfg.get('openai_api_base', 'https://api.openai.com/v1')
MAX_ENTRY_PER_FEED = cfg.get('max_entry', 100)
PATH_BASE = cfg.get('path_base', 'docs/')
LOG_FILE_PATH = f"{PATH_BASE}log/"

zhipuai.api_key = ZHIPUAI_API_KEY

feed_url_set_cache = {}
feed_token_cost = {}

log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    for c in cfg['feed']:
        name = c['name']
        feed_filename = f"{PATH_BASE}{name}.xml"
        feed_cache_filename = f"{PATH_BASE}cache/{name}.cache"
        feed_count = 0

        log_directory_path = f"{LOG_FILE_PATH}{name}"
        if not os.path.isdir(log_directory_path):
            # 如果目录不存在，使用 os 模块的 mkdir 函数创建目录
            os.mkdir(log_directory_path)

        log_file_handler = logging.FileHandler(f"{log_directory_path}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
        log_file_handler.setLevel(logging.INFO)
        log_file_handler.setFormatter(log_format)
        log_file_handler.set_name(name)
        logger.addHandler(log_file_handler)

        rss_feed = FeedGenerator()
        rss_feed.title(name)
        rss_feed.link(href='https://zturn.eu.org', rel='alternate')
        rss_feed.description(c.get('description', 'Powered by AI'))
        rss_urls = c['urls']

        # 抓取新文章
        # 读缓存
        cache = read_cache(feed_cache_filename)
        origin_cache = set(cache)
        feed_url_set_cache[name] = cache
        for url in rss_urls:
            logger.info(f"Parse: {url}")
            # 解析 url
            count = parse_url(c, rss_feed=rss_feed, url=url)
            feed_count += count
            logger.info(f"End: {url} 已解析 {count} 篇文章")

        # 保存新文章并排序
        entries = rss_feed.entry()
        sorted_list = sorted(entries, key=(lambda x: x.pubDate()), reverse=True)
        rss_feed.entry(sorted_list, replace=True)

        # 旧文章，总数不超过 max_entry_per_feed
        if os.path.exists(feed_filename):
            old_feed = feedparser.parse(feed_filename)
            for entry in reversed(old_feed.entries):
                if len(rss_feed.entry()) < MAX_ENTRY_PER_FEED:
                    add_entry_to_feed(rss_feed, entry, summarize=False, order='append')
                else:
                    break

        # 保存 rss 文件
        rss_feed.rss_file(feed_filename, pretty=True)
        # 更新缓存
        append_write_cache(feed_cache_filename, cache - origin_cache)

        # 输出统计信息
        logger.info(f"{name} 已新增 {feed_count} 篇")
        logger.info(f"{name} AI总结花费 {feed_token_cost.get(name, {})}")
        
        logger.removeHandler(log_file_handler)
        log_file_handler.close()

if __name__ == '__main__':
    main()
