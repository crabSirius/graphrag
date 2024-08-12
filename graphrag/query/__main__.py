# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Query Engine package root."""

import argparse
import logging
import logging.config
from enum import Enum

from .cli import run_global_search, run_local_search

INVALID_METHOD_ERROR = "Invalid method"


logging.config.fileConfig('logging.ini')
logger = logging.getLogger('graphrag')

class SearchType(Enum):
    """The type of search to run."""

    LOCAL = "local"
    GLOBAL = "global"

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        help="The configuration yaml file to use when running the query",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--data",
        help="The path with the output data from the pipeline",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--root",
        help="The data project root. Default value: the current directory",
        required=False,
        default=".",
        type=str,
    )

    parser.add_argument(
        "--method",
        help="The method to run, one of: local or global",
        required=True,
        type=SearchType,
        choices=list(SearchType),
    )

    parser.add_argument(
        "--community_level",
        help="Community level in the Leiden community hierarchy from which we will load the community reports higher value means we use reports on smaller communities",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--response_type",
        help="Free form text describing the response type and format, can be anything, e.g. Multiple Paragraphs, Single Paragraph, Single Sentence, List of 3-7 Points, Single Page, Multi-Page Report",
        type=str,
        default="Multiple Paragraphs",
    )

    parser.add_argument(
        "query",
        nargs=1,
        help="The query to run",
        type=str,
    )

    args = parser.parse_args()

    args.data = r'E:\codes\RAG\graphrag\ISV_prompt_tune_v2\output\20240806-193811\artifacts'
    args.data = r'E:\codes\RAG\graphrag\ISV_prompt_tune\output\20240729-171150\artifacts'
    args.data = r'E:\codes\RAG\graphrag\ISV_prompt_tune_v2\output\20240809-180652\artifacts'
    args.root = r'E:\codes\RAG\graphrag'
    args.community_level = 2
    args.method = SearchType.LOCAL
    args.query = [r'接入哪些平台需要收费，费用分别是多少?']
    # args.query = [r'广告平台接入了哪些？']
    # args.query = [r'预策魔方可以对接哪些广告平台']
    args.query = [r'预策数据魔方可以对接哪些平台？']
    match args.method:
        case SearchType.LOCAL:
            run_local_search(
                args.config,
                args.data,
                args.root,
                args.community_level,
                args.response_type,
                args.query[0],
            )
        case SearchType.GLOBAL:
            run_global_search(
                args.config,
                args.data,
                args.root,
                args.community_level,
                args.response_type,
                args.query[0],
            )
        case _:
            raise ValueError(INVALID_METHOD_ERROR)

    # args.query = [r'可以对接哪些平台', "广告平台接入了哪些", "接入平台的收费情况", "小红书有哪些接口可以提供",
    #               "淘宝万相台授权失败", "京东ISV订单已经接入，但是财务账单提示失败"]
    # method = [SearchType.GLOBAL]
    # for m in method:
    #     for q in args.query:
    #         if m == SearchType.LOCAL:
    #             run_local_search(
    #                 args.data,
    #                 args.root,
    #                 args.community_level,
    #                 args.response_type,
    #                 q,
    #             )
    #         elif m == SearchType.GLOBAL:
    #             run_global_search(
    #                 args.data,
    #                 args.root,
    #                 args.community_level,
    #                 args.response_type,
    #                 q,
    #             )
    #         else:
    #             raise ValueError(INVALID_METHOD_ERROR)
