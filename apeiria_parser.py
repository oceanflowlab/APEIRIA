from lark import Lark, Transformer, v_args
from lark import Tree

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Callable, Tuple, Set

from collections import Counter
import pretty_errors
import sys

LEGAL_FUNCTIONS = [
    'filter', 'scene', 'relate', 'relate_anchor', 'relate_multi',
    'intersection', 'exclude', 'union', 'intersect'
]

dsl_grammar = """
    start: function_call
    function_call: function_name "(" (argument ("," argument)*)? ")"
    function_name: NAME
    // argument: function_application | constant

    argument: concept
            | concept_set
            | function_call  // Allow nested function calls as arguments

    concept: NAME
    concept_set: "[" [concept ("," concept)*] "]"

    %import common.LETTER
    %import common.DIGIT
    NAME: (LETTER|DIGIT) ("_"|"-"|"'"|LETTER|DIGIT)*

    %import common.WS
    %ignore WS
"""

# Create the parser
# parser = Lark(dsl_grammar, start='start', parser='lalr')
PARSER = Lark(dsl_grammar, start='start', propagate_positions=True)

inline_args = v_args(inline=True)

# Define a Transformer to traverse the tree and execute the function calls
class ExecuteDSL(Transformer):
    def __init__(self, verbose=False, input_text=None):
        self.verbose = verbose
        self.input_text = input_text
        self.indent = 0  # For pretty printing the DFS traversal

    def _print(self, message):
        if self.verbose:
            print("  " * self.indent + message)

    # def _get_original_text(self, node_or_nodes):
    #     assert self.input_text is not None, "Input text is required to get the original text of a node"

    #     if not isinstance(node_or_nodes, list):
    #         node_or_nodes = [node_or_nodes]

    #     return self.input_text[node_or_nodes[0].start_pos:node_or_nodes[-1].end_pos+1]

    def _get_original_text(self, node_or_nodes):
        assert self.input_text is not None, "Input text is required to get the original text of a node"

        if not isinstance(node_or_nodes, list):
            node_or_nodes = [node_or_nodes]

        # Handle nested lists recursively
        def _get_start_and_end(node):
            if isinstance(node, list):
                start_0, end_0 = _get_start_and_end(node[0])
                start_1, end_1 = _get_start_and_end(node[-1])
                return start_0, end_1

            elif isinstance(node, dict):
                print(f"warning: dict node: {node}", file=sys.stderr)
                return node['start_pos'], node['end_pos']

            else:
                return node.start_pos, node.end_pos

        start_pos, end_pos = _get_start_and_end(node_or_nodes)

        return self.input_text[start_pos:end_pos+1]

    def start(self, args):
        self._print("Starting DFS traversal")
        return args[0]

    @inline_args
    def function_call(self, function_name, *arguments):
        # Simulate executing the function
        # print(function_name, arguments)
        self._print(f"Function call: {self._get_original_text([function_name] + list(arguments))}")
        self.indent += 1
        result = self.execute_function(function_name, arguments)
        self.indent -= 1

        return result

    def function_name(self, args):
        fn_name = args[0]#.value
        # self._print(f"Function name: {fn_name}")
        return fn_name

    def concept(self, args):
        concept_name = args[0]#.value
        # self._print(f"Concept: {concept_name}")
        return concept_name

    def argument(self, args):
        # self._print(f"Argument: {args[0]}")
        return args[0]

    def concept_set(self, args):
        # self._print(f"Concept Set: {args}")
        return args

    def execute_function(self, function_name, arguments):
        # Simulate function execution
        arguments_string = ', '.join(map(str, arguments))
        self._print(f"Executing: {function_name} with arguments: ({arguments})")
        # Here you would implement the actual logic for each function call

        # start_pos = function_name.start_pos
        # end_pos = arguments[-1].end_pos if len(arguments) > 0 else function_name.end_pos
        # function_call_repr = self.input_text[start_pos:end_pos]
        # self._print(f"Original text: {function_call_repr}")

        return f"Result of {function_name}({arguments_string})"


def parse_program_string(program_string, parser=PARSER):
    # Parse the input
    parse_tree = parser.parse(program_string)
    return parse_tree

def is_valid_program(program: str, parser=PARSER) -> bool | Tree:
    try:
        tree = parser.parse(program)
        return tree
    except Exception as e:
        return False

def extract_function_names(parse_tree: Tree) -> list[str]:
    function_names = []

    def traverse(node):
        if isinstance(node, Tree):
            if node.data == 'function_call':
                function_names.append(node.children[0].children[0])
            for child in node.children:
                traverse(child)

    traverse(parse_tree)
    return function_names

def is_legal_program(tree: Tree, legal_functions=LEGAL_FUNCTIONS) -> bool:
    function_names = extract_function_names(tree)
    return set(function_names).issubset(legal_functions)

class OutputType(Enum):
    OBJECT_SET = "object_set"
    STRING = "string"

class FunctionType(Enum):
    DIRECT = "direct"
    NORMAL = "normal"

@dataclass
class FunctionSignature:
    name: str
    args_types: List[str]
    args_names: List[str]
    output_type: OutputType
    function_type: FunctionType = FunctionType.NORMAL
    direct_implementation: Optional[Callable] = None
    variadic: bool = False  # New field to indicate variadic arguments
    variadic_type: Optional[str] = None  # Type of variadic arguments


class TypeChecker:
    def __init__(self, signatures: Dict[str, FunctionSignature]):
        self.signatures = signatures
        self.current_scope = {}

    def get_node_type(self, node: Tree) -> str:
        if node.data == 'function_call':
            fn_name = node.children[0].children[0]
            if fn_name not in self.signatures:
                raise TypeError(f"Unknown function: {fn_name}")
            return self.signatures[fn_name].output_type.value

        elif node.data == 'concept':
            return "string"

        elif node.data == 'concept_set':
            return "object_set"

        elif node.data == 'argument':
            return self.get_node_type(node.children[0])

        raise TypeError(f"Cannot determine type for node: {node.data}")

    def is_function_call(self, node: Tree) -> bool:
        if node.data == 'function_call':
            return True

        elif node.data == 'argument':
            return self.is_function_call(node.children[0])

        elif node.data == 'concept':
            return False

        elif node.data == 'concept_set':
            return False

        raise TypeError(f"Cannot determine if node is a function call: {node.data}")


    def check_types(self, tree: Tree) -> bool:
        if tree.data == 'function_call':
            fn_name = tree.children[0].children[0]
            signature = self.signatures[fn_name]
            arguments = tree.children[1:]  # Skip function name

            # Handle variadic functions
            if signature.variadic:
                if len(arguments) < 1:  # Require at least one argument for variadic functions
                    raise TypeError(
                        f"Function {fn_name} expects at least 1 argument of type {signature.variadic_type}"
                    )

                # Check all arguments are of the variadic type
                for i, arg in enumerate(arguments):
                    arg_type = self.get_node_type(arg)
                    if arg_type != signature.variadic_type:
                        raise TypeError(
                            f"Function {fn_name}'s argument {i+1} expects {signature.variadic_type}, "
                            f"but got {arg_type}"
                        )


            else:
                # Normal function type checking
                if len(arguments) != len(signature.args_types):
                    raise TypeError(
                        f"Function {fn_name} expects {len(signature.args_types)} arguments "
                        f"({', '.join(signature.args_names)}), but got {len(arguments)}"
                    )

                # Check each argument's type
                for arg, expected_type, arg_name in zip(
                    arguments,
                    signature.args_types,
                    signature.args_names
                ):
                    arg_type = self.get_node_type(arg)
                    if arg_type != expected_type:
                        raise TypeError(
                            f"Function {fn_name}'s argument '{arg_name}' expects {expected_type}, "
                            f"but got {arg_type}"
                        )

            # Recursively check nested function calls
            for child in arguments:
                # if isinstance(child, Tree) and child.data == 'function_call':
                #     self.check_types(child)
                if self.is_function_call(child):
                    self.check_types(child)

            return True

        else:
            # Recursively check nested function calls
            for child in tree.children:
                self.check_types(child)

        return True

def get_default_signatures() -> Dict[str, FunctionSignature]:
    signatures = {
        "scene": FunctionSignature(
            name="scene",
            args_types=[],
            args_names=[],
            output_type=OutputType.OBJECT_SET,
            function_type=FunctionType.DIRECT,
        ),
        "filter": FunctionSignature(
            name="filter",
            args_types=["object_set", "string"],
            args_names=["objects", "attribute"],
            output_type=OutputType.OBJECT_SET
        ),
        "relate": FunctionSignature(
            name="relate",
            args_types=["object_set", "object_set", "string"],
            args_names=["target_objects", "objects_2", "relation"],
            output_type=OutputType.OBJECT_SET
        ),
        "relate_anchor": FunctionSignature(
            name="relate_anchor",
            args_types=["object_set", "object_set", "object_set", "string"],
            args_names=["target_objects", "objects_2", "anchor_objects", "relation"],
            output_type=OutputType.OBJECT_SET
        ),
        "relate_multi": FunctionSignature(
            name="relate_multi",
            args_types=["object_set", "object_set", "object_set", "string"],
            args_names=["target_objects", "objects_2", "objects_3", "relation"],
            output_type=OutputType.OBJECT_SET
        ),
        "query_attribute": FunctionSignature(
            name="query_attribute",
            args_types=["object_set", "string"],
            args_names=["objects", "attribute"],
            output_type=OutputType.STRING
        ),
        "find": FunctionSignature(
            name="find",
            args_types=["string"],
            args_names=["description"],
            output_type=OutputType.OBJECT_SET
        ),
        "intersection": FunctionSignature(
            name="intersection",
            args_types=[],  # Empty for variadic functions
            args_names=[],  # Empty for variadic functions
            output_type=OutputType.OBJECT_SET,
            function_type=FunctionType.DIRECT,
            variadic=True,
            variadic_type="object_set"
        ),
        "union": FunctionSignature(
            name="union",
            args_types=[],  # Empty for variadic functions
            args_names=[],  # Empty for variadic functions
            output_type=OutputType.OBJECT_SET,
            function_type=FunctionType.DIRECT,
            variadic=True,
            variadic_type="object_set"
        ),
        "exclude": FunctionSignature(
            name="exclude",
            args_types=["object_set", "object_set"],
            args_names=["objects_1", "objects_2"],
            output_type=OutputType.OBJECT_SET,
            function_type=FunctionType.DIRECT,
        ),
        "count": FunctionSignature(
            name="count",
            args_types=["object_set", "string"],
            args_names=["objects", "attribute"],
            output_type=OutputType.STRING
        ),
    }

    # aliases
    signatures["intersect"] = signatures["intersection"]

    return signatures

def get_grounding_signatures() -> Dict[str, FunctionSignature]:
    signatures = get_default_signatures()

    # remove count, query_attribute, find
    del signatures["count"]
    del signatures["query_attribute"]
    del signatures["find"]

    return signatures

def is_type_valid_program(tree: Tree, signatures: Dict[str, FunctionSignature]) -> bool:
    try:
        type_checker = TypeChecker(signatures)
        return type_checker.check_types(tree)
    except TypeError as e:
        print(f"Type error: {e}")
        return False

def is_valid_legal_type_valid_program(
    program: str,
    parser=PARSER,
    signatures: Dict[str, FunctionSignature] = None
) -> bool | Tree:
    if signatures is None:
        signatures = get_default_signatures()

    try:
        tree = parser.parse(program)
        if not is_legal_program(tree, legal_functions=list(signatures.keys())):
            print("Illegal function call")
            return False
        if not is_type_valid_program(tree, signatures):
            return False
        return tree
    except Exception as e:
        print(f"Parse error: {e}")
        return False

def get_call_depth(tree: Tree) -> Tuple[int, List[str]]:
    """
    Analyze the maximum nesting depth of function calls and return the deepest path.
    Returns a tuple of (max_depth, path_of_deepest_calls)
    """
    max_depth = 0
    max_path = []

    def traverse(node: Tree, current_depth: int = 0, current_path: List[str] = None) -> None:
        nonlocal max_depth, max_path

        if current_path is None:
            current_path = []

        if isinstance(node, Tree):
            if node.data == 'function_call':
                fn_name = node.children[0].children[0]
                current_depth += 1
                current_path.append(fn_name)

                if current_depth > max_depth:
                    max_depth = current_depth
                    max_path = current_path.copy()

                for child in node.children[1:]:  # Skip function name
                    traverse(child, current_depth, current_path.copy())

                current_path.pop()
            else:
                for child in node.children:
                    traverse(child, current_depth, current_path.copy())

    traverse(tree)
    return max_depth, max_path

def analyze_function_calls(tree: Tree, exclude: set[str] = {'scene'}) -> Dict[str, int]:
    """
    Analyze the function calls in a parse tree and return a counter of function usage.

    Args:
        tree: The parse tree to analyze
        exclude: Set of function names to exclude from the count (default: {'scene'})

    Returns:
        Dict mapping function names to their call count
    """
    counter = Counter()

    def traverse(node: Tree):
        if isinstance(node, Tree):
            if node.data == 'function_call':
                fn_name = node.children[0].children[0]
                if fn_name not in exclude:
                    counter[fn_name] += 1
            for child in node.children:
                traverse(child)

    traverse(tree)
    return dict(counter)

def extract_attributes(tree: Tree, signatures: Dict[str, FunctionSignature]) -> Dict[str, Counter]:
    """
    Extract attributes used in different function contexts.

    Returns:
        Dict mapping function types to Counters of attributes used. For example:
        {
            'unary_attributes': Counter({'red': 2, 'large': 1}),  # from filter()
            'binary_attributes': Counter({'on': 1}),  # from relate()
            'ternary_attributes': Counter({'between': 1}),  # from relate_anchor/multi
            'query_attributes': Counter({'color': 1})
        }
    """
    attributes = {
        'unary_attributes': Counter(),    # filter attributes
        'binary_attributes': Counter(),   # relate attributes
        'ternary_attributes': Counter(),  # relate_anchor/multi attributes
        'query_attributes': Counter()     # query_attribute/count attributes
    }

    def traverse(node: Tree):
        if isinstance(node, Tree):
            if node.data == 'function_call':
                fn_name = node.children[0].children[0]
                # print(fn_name)

                # Get the function's signature
                if fn_name not in signatures:
                    return

                sig = signatures[fn_name]

                # Extract attributes based on function type
                if fn_name == 'filter':
                    # Second argument is the attribute
                    if len(node.children) >= 3:
                        attr_node = node.children[2].children[0]
                        # print(attr_node)
                        if attr_node.data == 'concept':
                            attributes['unary_attributes'][attr_node.children[0]] += 1

                elif fn_name == 'relate':
                    # Last argument is the relation
                    if len(node.children) >= 2:
                        attr_node = node.children[-1].children[0]
                        if attr_node.data == 'concept':
                            attributes['binary_attributes'][attr_node.children[0]] += 1

                elif fn_name in {'relate_anchor', 'relate_multi'}:
                    # Last argument is the relation
                    if len(node.children) >= 2:
                        attr_node = node.children[-1].children[0]
                        if attr_node.data == 'concept':
                            attributes['ternary_attributes'][attr_node.children[0]] += 1

                elif fn_name in {'query_attribute', 'count'}:
                    # Second argument is the attribute
                    if len(node.children) >= 3:
                        attr_node = node.children[2].children[0]
                        if attr_node.data == 'concept':
                            attributes['query_attributes'][attr_node.children[0]] += 1

            # Recursively process children
            for child in node.children:
                traverse(child)

    traverse(tree)
    return attributes

def analyze_programs_attributes(programs: List[str], signatures: Dict[str, FunctionSignature] = None) -> Dict[str, Counter]:
    """
    Analyze multiple programs for attribute usage and return frequency counts.

    Returns:
        Dict mapping attribute types to Counters of their frequencies
    """
    if signatures is None:
        signatures = get_default_signatures()

    total_counters = {
        'unary_attributes': Counter(),
        'binary_attributes': Counter(),
        'ternary_attributes': Counter(),
        'query_attributes': Counter()
    }

    for program in programs:
        tree = is_valid_legal_type_valid_program(program, signatures=signatures)
        if tree:
            attributes = extract_attributes(tree, signatures)
            for attr_type, counter in attributes.items():
                total_counters[attr_type].update(counter)

    return total_counters



if __name__ == "__main__":

    # Example input with nested function calls and object sets
    input_text = "relate(filter(scene(), APPLE), filter(scene(), red), on)"
    parse_tree = parse_program_string(input_text)
    print(parse_tree.pretty())

    transformer = ExecuteDSL(verbose=True, input_text=input_text)
    result = transformer.transform(parse_tree)
    # Print the final result
    print("\nFinal result:", result)

    # --- Test the type checker ---
    signatures = get_default_signatures()

    test_programs = [
        # Basic function calls
        ("scene()", True, "Basic scene call"),
        ("filter(scene(), red)", True, "Basic filter call"),

        # Nested function calls
        ("filter(filter(scene(), red), large)", True, "Nested filters"),
        ("relate(filter(scene(), apple), filter(scene(), table), on)", True, "Complex nested call"),

        # Variadic functions
        ("intersection(scene())", True, "Intersection with one argument"),
        ("intersection(scene(), filter(scene(), red))", True, "Intersection with two arguments"),
        ("intersection(scene(), filter(scene(), red), filter(scene(), large))", True, "Intersection with three arguments"),
        ("union(scene())", True, "Union with one argument"),
        ("union(scene(), scene(), scene())", True, "Union with three arguments"),
        ("intersect(scene(), scene())", True, "Intersection alias"),

        # Relate variations
        ("relate(scene(), scene(), on)", True, "Basic relate"),
        ("relate_anchor(scene(), scene(), scene(), between)", True, "Relate anchor"),
        ("relate_multi(scene(), scene(), scene(), aligned)", True, "Relate multi"),
        ("relate()", False, "Relate with no arguments"),
        ("relate(scene(), scene())", False, "Relate with missing argument"),
        ("relate(scene(), scene(), scene(), scene())", False, "Relate with extra argument"),

        # Query and Find
        ("query_attribute(scene(), color)", True, "Query attribute"),
        ("find(large_red_apple)", True, "Find object"),
        ("count(scene(), apples)", True, "Count objects"),

        # Error cases - Invalid number of arguments
        ("scene(scene())", False, "Scene with arguments"),
        ("filter(scene())", False, "Filter missing argument"),
        ("relate(scene(), on)", False, "Relate missing argument"),
        ("intersection()", False, "Empty intersection"),
        ("relate_anchor(scene(), scene(), between)", False, "Relate anchor missing argument"),

        # Error cases - Invalid argument types
        ("filter(red, scene())", False, "Filter with wrong argument order"),
        ("relate(scene(), table, scene())", False, "Relate with wrong argument type"),
        ("intersection(apple, orange)", False, "Intersection with wrong types"),
        ("union(scene(), red)", False, "Union with invalid argument type"),
        ("query_attribute(color, scene())", False, "Query attribute wrong order"),

        # Complex nested cases
        ("relate(filter(scene(), apple), intersection(filter(scene(), table), filter(scene(), wooden)), on)",
         True, "Complex nested with intersection"),
        ("union(filter(scene(), red), filter(scene(), blue), intersection(scene(), filter(scene(), large)))",
         True, "Complex nested with union and intersection"),
        ("relate_anchor(filter(scene(), book), filter(scene(), shelf), union(scene(), filter(scene(), wall)), between)",
         True, "Complex relate_anchor with union"),

        # Edge cases
        ("exclude(scene(), scene())", True, "Basic exclude"),
        ("exclude(intersection(scene(), scene()), union(scene(), scene()))", True, "Complex exclude"),
        ("filter(exclude(scene(), filter(scene(), red)), large)", True, "Nested exclude"),

        # Invalid function names
        ("unknown_function(scene())", False, "Non-existent function"),
        ("SCENE()", False, "Case sensitive function name"),

        # Complex error cases
        ("relate(filter(scene(), apple), intersection(), on)", False, "Empty intersection in nested call"),
        ("union(filter(scene(), red), relate())", False, "Invalid nested function call"),

        # Additional attribute combinations
        ("filter(scene(), dark_red)", True, "Compound attribute"),
        ("filter(scene(), very_large_wooden)", True, "Multiple word attribute"),
        ("relate(scene(), scene(), directly_above)", True, "Complex relation"),

        # Multiple operations on same level
        ("intersection(filter(scene(), red), filter(scene(), round), filter(scene(), small))",
         True, "Multiple filters in intersection"),
        ("union(filter(scene(), wooden), filter(scene(), metal), filter(scene(), plastic))",
         True, "Multiple filters in union"),
    ]

    for program, should_pass, description in test_programs:
        print(f"\nTesting: {description}")
        print(f"Program: {program}")
        result = is_valid_legal_type_valid_program(program, signatures=signatures)
        passed = bool(result) == should_pass
        print(f"{'✓ PASS' if passed else '✗ FAIL'}: Expected {'valid' if should_pass else 'invalid'}, "
              f"got {'valid' if bool(result) else 'invalid'}")

    # --- Test the call depth analyzer ---
    input_text = "relate(filter(scene(), apple), filter(scene(), red), on)"
    parse_tree = parse_program_string(input_text)
    print(parse_tree.pretty())
    max_depth, max_path = get_call_depth(parse_tree)

    print(f"Max depth: {max_depth}")
    print(f"Path of deepest calls: {max_path}")

    # --- Test the function call analysis ---
    analysis_programs = [
        (
            "relate(filter(scene(), apple), intersection(filter(scene(), table), filter(scene(), wooden)), on)",
            "Complex nested with intersection"
        ),
        (
            "union(filter(scene(), red), filter(scene(), blue), intersection(scene(), filter(scene(), large)))",
            "Complex nested with union and intersection"
        ),
        (
            "relate_anchor(filter(scene(), book), filter(scene(), shelf), union(scene(), filter(scene(), wall)), between)",
            "Complex relate_anchor with union"
        ),
    ]

    print("\nFunction Call Analysis:")
    print("=" * 50)

    for program, description in analysis_programs:
        print(f"\nAnalyzing: {description}")
        print(f"Program: {program}")

        tree = is_valid_legal_type_valid_program(program, signatures=signatures)
        if tree:
            stats = analyze_function_calls(tree, exclude={'scene', 'intersection', 'union'})
            print("\nFunction call counts:")
            for func, count in sorted(stats.items()):
                print(f"  {func}: {count} call{'s' if count > 1 else ''}")
            print(f"Total (excluding scene): {sum(stats.values())} calls")
        else:
            print("Invalid program - could not analyze")
        print("-" * 50)


    # Example usage with more diverse test cases:
    test_programs = [
        "filter(scene(), red)",
        "relate(filter(scene(), apple), filter(scene(), table), on)",
        "query_attribute(filter(scene(), large), color)",
        "relate_anchor(filter(scene(), book), scene(), filter(scene(), wall), between)",
        "relate_multi(filter(scene(), cup), filter(scene(), plate), filter(scene(), table), aligned)",
        "filter(scene(), large)",
        "relate(scene(), filter(scene(), chair), next_to)",
        "relate_anchor(scene(), scene(), filter(scene(), wall), against)",
    ]

    # print program and analyze
    for program in test_programs:
        print(f"\nAnalyzing program: {program}")
        results = analyze_programs_attributes([program])
        print("\nAttribute usage analysis:")
        for attr_type, counter in results.items():
            if counter:
                print(f"\n{attr_type}:")
                for attr, count in counter.most_common():
                    print(f"  {attr}: {count}")

    # Analyze programs
    results = analyze_programs_attributes(test_programs)
    print("\nAttribute usage analysis:")
    for attr_type, counter in results.items():
        if counter:
            print(f"\n{attr_type}:")
            for attr, count in counter.most_common():
                print(f"  {attr}: {count}")
