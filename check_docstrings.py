import ast
import os

def has_docstring(node):
    """Check if a function/class has a docstring."""
    return (ast.get_docstring(node) is not None)

stats = {}
for filename in os.listdir('functions'):
    if filename.endswith('.py'):
        filepath = os.path.join('functions', filename)
        try:
            with open(filepath) as f:
                tree = ast.parse(f.read(), filename=filename)
            
            funcs_with_docs = 0
            funcs_without_docs = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if has_docstring(node):
                        funcs_with_docs += 1
                    else:
                        funcs_without_docs += 1
            
            if funcs_with_docs + funcs_without_docs > 0:
                stats[filename] = {
                    'with': funcs_with_docs,
                    'without': funcs_without_docs,
                    'total': funcs_with_docs + funcs_without_docs
                }
        except SyntaxError as e:
            print(f"Syntax error in {filename}: {e}")

# Sort by number without docstrings
sorted_stats = sorted(stats.items(), key=lambda x: x[1]['without'], reverse=True)

print("\nFunctions by file (sorted by functions needing docstrings):")
print(f"{'File':<40} {'With Docs':<12} {'Without Docs':<15} {'Total':<8}")
print("-" * 80)

total_with = 0
total_without = 0
for filename, data in sorted_stats:
    print(f"{filename:<40} {data['with']:<12} {data['without']:<15} {data['total']:<8}")
    total_with += data['with']
    total_without += data['without']

print("-" * 80)
print(f"{'TOTAL':<40} {total_with:<12} {total_without:<15} {total_with + total_without:<8}")
print(f"\nPercentage with docstrings: {100 * total_with / (total_with + total_without):.1f}%")
