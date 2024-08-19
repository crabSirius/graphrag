# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'GraphExtractionResult' and 'GraphExtractor' models."""

import logging
import numbers
import re
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx
import tiktoken

import graphrag.config.defaults as defs
from graphrag.index.typing import ErrorHandlerFn
from graphrag.index.utils import clean_str
from graphrag.llm import CompletionLLM

from .prompts import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT, LOOP_PROMPT, ADDITIONAL_PROMPT

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    output: nx.Graph
    source_docs: dict[Any, Any]


class GraphExtractor:
    """Unipartite graph extractor class definition."""

    _llm: CompletionLLM
    _join_descriptions: bool
    _tuple_delimiter_key: str
    _record_delimiter_key: str
    _entity_types_key: str
    _input_text_key: str
    _completion_delimiter_key: str
    _entity_name_key: str
    _input_descriptions_key: str
    _extraction_prompt: str
    _summarization_prompt: str
    _loop_args: dict[str, Any]
    _max_gleanings: int
    _on_error: ErrorHandlerFn

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        tuple_delimiter_key: str | None = None,
        record_delimiter_key: str | None = None,
        input_text_key: str | None = None,
        entity_types_key: str | None = None,
        completion_delimiter_key: str | None = None,
        prompt: str | None = None,
        join_descriptions=True,
        encoding_model: str | None = None,
        max_gleanings: int | None = None,
        on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        # TODO: streamline construction
        self._llm = llm_invoker
        self._join_descriptions = join_descriptions
        self._input_text_key = input_text_key or "input_text"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        self._entity_types_key = entity_types_key or "entity_types"
        self._extraction_prompt = prompt or GRAPH_EXTRACTION_PROMPT
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else defs.ENTITY_EXTRACTION_MAX_GLEANINGS
        )
        self._on_error = on_error or (lambda _e, _s, _d: None)

        # Construct the looping arguments
        encoding = tiktoken.get_encoding(encoding_model or "cl100k_base")
        yes = encoding.encode("YES")
        no = encoding.encode("NO")
        self._loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}
        self._source_text_cache = {}
        self._synonyms_map_dict = {
            '预策数据魔方': ['魔方', '数据魔方', '魔方平台'],
            '订单信息': ['订单', '订单信息', '订单列表信息'],
            '销售订单': ['销售订单', '销售订单信息'],
            'APP KEY': ['APP KEY', 'APP_KEY', 'APPKEY'],
            'APP SECRET': ['APP SECRET', 'APP_SECRET', 'APPSECRET'],
            '商家自研': ['商家自研', '商家自研模式'],
            'ISV': ['ISV', 'ISV模式', 'ISV对接', 'ISV接入方式'],
            '无费用': ['无', '无费用', '无需订购'],
            '预策ERP服务': ['预策ERP服务', '预策ERP'],
            '快手-磁力金牛': ['磁力金牛', '快手-磁力金牛', '快手-磁力金牛平台', '快手磁力金牛开放平台'],
            '聚水潭开放平台': ['聚水潭', '聚水潭开放平台'],
            '班牛开放平台': ['班牛开放平台', '班牛平台', '班牛'],
        }
        self._relation_map_dict = {
            '旺店通开放平台':
                {
                    'relation': {'source': '预策数据魔方', 'target': '旺店通开放平台', 'relation_description': '预策数据魔方支持接入旺店通开放平台'},
                    'done_flag': False
                },
            '宜搭':
                {
                    'relation': {'source': '预策数据魔方', 'target': '宜搭',
                                 'relation_description': '预策数据魔方支持接入钉钉宜搭平台'},
                    'done_flag': False
                },
            '班牛开放平台':
                {
                    'relation': {'source': '预策数据魔方', 'target': '班牛开放平台',
                                 'relation_description': '预策数据魔方支持接入班牛开放平台'},
                    'done_flag': False
                },
        }
        self._additional_entities = {
            '巨量千川-PC': [],
            '巨量千川-随心推': ['预策数据魔方', '巨量千川-随心推'],
        }

    async def __call__(
        self, texts: list[str], prompt_variables: dict[str, Any] | None = None
    ) -> GraphExtractionResult:
        """Call method definition."""
        if prompt_variables is None:
            prompt_variables = {}
        all_records: dict[int, str] = {}
        source_doc_map: dict[int, str] = {}

        # Wire defaults into the prompt variables
        prompt_variables = {
            **prompt_variables,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or DEFAULT_COMPLETION_DELIMITER,
            self._entity_types_key: ",".join(
                prompt_variables[self._entity_types_key] or DEFAULT_ENTITY_TYPES
            ),
        }

        for doc_index, text in enumerate(texts):
            try:
                # Invoke the entity extraction
                result = await self._process_document(text, prompt_variables)
                source_doc_map[doc_index] = text
                all_records[doc_index] = result
            except Exception as e:
                logging.exception("error extracting graph")
                self._on_error(
                    e,
                    traceback.format_exc(),
                    {
                        "doc_index": doc_index,
                        "text": text,
                    },
                )

        output = await self._process_results(
            all_records,
            source_doc_map,
            prompt_variables.get(self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER),
            prompt_variables.get(self._record_delimiter_key, DEFAULT_RECORD_DELIMITER),
        )

        return GraphExtractionResult(
            output=output,
            source_docs=source_doc_map,
        )

    def _check_additional_entities(self, results, text):
        # todo 优化_add_entities与对应chunk匹配的判断
        entities = []
        relationships = []
        for entity in self._additional_entities:
            if entity in text and entity not in results:
                entities.append(entity)
                if len(self._additional_entities[entity]) == 2:
                    relationships.append("->".join(self._additional_entities[entity]))
        return entities, relationships

    async def _process_document(
        self, text: str, prompt_variables: dict[str, str]
    ) -> str:
        response = await self._llm(
            self._extraction_prompt,
            variables={
                **prompt_variables,
                self._input_text_key: text,
            },
        )
        results = response.output or ""

        # Repeat to ensure we maximize entity count
        for i in range(self._max_gleanings):
            response = await self._llm(
                CONTINUE_PROMPT,
                name=f"extract-continuation-{i}",
                history=response.history,
            )
            results += response.output or ""

            # if this is the final glean, don't bother updating the continuation flag
            if i >= self._max_gleanings - 1:
                break

            response = await self._llm(
                LOOP_PROMPT,
                name=f"extract-loopcheck-{i}",
                history=response.history,
                model_parameters=self._loop_args,
            )
            if response.output != "YES":
                break
        additional_entities, additional_relationships = self._check_additional_entities(results, text)
        if len(additional_entities) > 0:
            additional_entities = '"、"'.join(additional_entities).join(['"', '"'])
            additional_relationships = '"、"'.join(additional_relationships).join(['"', '"'])
            response = await self._llm(
                ADDITIONAL_PROMPT,
                name=f"additional_entities",
                variables={
                    "additional_entities": additional_entities,
                    "additional_relationships": additional_relationships
                },
                history=response.history,
            )
            results += response.output or ""

        return results

    def _entity_is_in_text(self, entity_name_str: str, text: str, text_id: str):
        """Check if an entity is present in a given text."""
        if text_id in self._source_text_cache:
            new_text = self._source_text_cache[text_id]
        else:
            new_text = text.strip().upper().replace(" ", "").replace("_", "").replace("-", "")
            self._source_text_cache[text_id] = new_text
        entity_name_str = entity_name_str.strip().upper().replace(" ", "").replace("_", "").replace("-", "")
        match = entity_name_str in new_text
        if not match:
            print(f"Entity {entity_name_str} not found in text {text}")
        return match

    def _synonyms_map(self, entity_name):
        for k, v in self._synonyms_map_dict.items():
            if entity_name in v:
                return k
        return entity_name

    def _add_relations(self, graph, source_doc_id):
        for k, v in self._relation_map_dict.items():
            flag = v['done_flag']
            if flag:
                continue
            source = v['relation']['source']
            target = v['relation']['target']
            relation_description = v['relation']['relation_description']
            edge_source_id = clean_str(str(source_doc_id))
            if k in graph.nodes:
                if not graph.has_edge(source, target):
                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    graph.add_edge(
                        source,
                        target,
                        weight=1.0,
                        description=relation_description,
                        source_id=edge_source_id,
                    )
                    self._relation_map_dict[k]['done_flag'] = True

    async def _process_results(
        self,
        results: dict[int, str],
        src_doc: dict[int, str],
        tuple_delimiter: str,
        record_delimiter: str,
    ) -> nx.Graph:
        """Parse the result string to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain
            - tuple_delimiter - delimiter between tuples in an output record, default is '<|>'
            - record_delimiter - delimiter between records, default is '##'
        Returns:
            - output - unipartite graph in graphML format
        """

        graph = nx.Graph()
        for source_doc_id, extracted_data in results.items():
            records = [r.strip() for r in extracted_data.split(record_delimiter)]
            source_text = src_doc[source_doc_id]

            for record in records:
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(tuple_delimiter)

                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = clean_str(record_attributes[1].upper())
                    entity_name = self._synonyms_map(entity_name)
                    entity_type = clean_str(record_attributes[2].upper())
                    entity_description = clean_str(record_attributes[3])

                    if not self._entity_is_in_text(entity_name, source_text, source_doc_id):
                        print(entity_name)
                        print(source_text)

                    if entity_name in graph.nodes():
                        node = graph.nodes[entity_name]
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list({
                                    *_unpack_descriptions(node),
                                    entity_description,
                                })
                            )
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        node["source_id"] = ", ".join(
                            list({
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            })
                        )
                        node["entity_type"] = (
                            entity_type if entity_type != "" else node["entity_type"]
                        )
                    else:
                        graph.add_node(
                            entity_name,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )

                if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 5
                ):
                    # add this record as edge
                    source = clean_str(record_attributes[1].upper())
                    target = clean_str(record_attributes[2].upper())
                    source = self._synonyms_map(source)
                    target = self._synonyms_map(target)
                    if not self._entity_is_in_text(source, source_text, source_doc_id):
                        print(source)
                        print(source_text)
                    if not self._entity_is_in_text(target, source_text, source_doc_id):
                        print(target)
                        print(source_text)
                    edge_description = clean_str(record_attributes[3])
                    edge_source_id = clean_str(str(source_doc_id))
                    weight = (
                        float(record_attributes[-1])
                        if isinstance(record_attributes[-1], numbers.Number)
                        else 1.0
                    )
                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                edge_description = "\n".join(
                                    list({
                                        *_unpack_descriptions(edge_data),
                                        edge_description,
                                    })
                                )
                            edge_source_id = ", ".join(
                                list({
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                })
                            )
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )
            self._add_relations(graph, source_doc_id)
        return graph


def _unpack_descriptions(data: Mapping) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data: Mapping) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")
