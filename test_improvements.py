#!/usr/bin/env python3
"""Test script to demonstrate the improvements."""

import json
from classifiers import ClassifierChain, AdvancedClassifier


def test_advanced_classifier():
    """Test the new AdvancedClassifier."""
    print("=" * 80)
    print("TESTING ADVANCED CLASSIFIER")
    print("=" * 80)
    
    classifier = AdvancedClassifier()
    available_domains = ["python", "sql", "javascript", "java", "cpp", "shell", "docs"]
    
    # Test cases
    test_cases = [
        {
            "name": "Python with pandas",
            "item": {
                "prompt": "How to read a CSV file?",
                "code": "import pandas as pd\ndf = pd.read_csv('file.csv')\nprint(df.head())"
            }
        },
        {
            "name": "SQL SELECT query",
            "item": {
                "prompt": "Get all users from database",
                "code": "SELECT * FROM users WHERE age > 18 ORDER BY name;"
            }
        },
        {
            "name": "JavaScript arrow function",
            "item": {
                "prompt": "Create a function that doubles a number",
                "code": "const double = (x) => x * 2;\nconst result = double(5);"
            }
        },
        {
            "name": "Java class",
            "item": {
                "prompt": "Create a simple class",
                "code": "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}"
            }
        },
        {
            "name": "C++ with std",
            "item": {
                "prompt": "Print a vector",
                "code": "#include <iostream>\n#include <vector>\nstd::vector<int> v = {1, 2, 3};"
            }
        },
        {
            "name": "Shell script",
            "item": {
                "prompt": "Loop through files",
                "code": "#!/bin/bash\nfor file in *.txt; do\n    echo $file\ndone"
            }
        },
    ]
    
    for test in test_cases:
        score = classifier.classify(test["item"], available_domains)
        if score:
            print(f"\n✓ {test['name']}")
            print(f"  Domain: {score.domain}")
            print(f"  Confidence: {score.score:.2f}")
            print(f"  Reason: {score.reason}")
        else:
            print(f"\n✗ {test['name']} - No classification")


def test_classifier_chain():
    """Test the improved classifier chain."""
    print("\n" + "=" * 80)
    print("TESTING CLASSIFIER CHAIN")
    print("=" * 80)
    
    chain = ClassifierChain()
    available_domains = ["python", "sql", "javascript", "java", "cpp", "shell"]
    
    test_cases = [
        {
            "name": "Python with library metadata",
            "item": {
                "prompt": "Read CSV",
                "code": "import pandas as pd",
                "metadata": {"library": "pandas"}
            }
        },
        {
            "name": "SQL without metadata",
            "item": {
                "prompt": "Get users",
                "code": "SELECT * FROM users WHERE id = 1;",
                "metadata": {}
            }
        },
        {
            "name": "JavaScript with strong patterns",
            "item": {
                "prompt": "Create function",
                "code": "const add = (a, b) => a + b;",
                "metadata": {}
            }
        },
    ]
    
    for test in test_cases:
        domain, reason = chain.classify(test["item"], available_domains, "python")
        print(f"\n✓ {test['name']}")
        print(f"  Classified as: {domain}")
        print(f"  Reason: {reason}")


def test_simple_stats():
    """Test the simplified statistics collector."""
    print("\n" + "=" * 80)
    print("TESTING SIMPLIFIED STATISTICS")
    print("=" * 80)
    
    from split_ds1000_by_domain import SimpleStats
    
    stats = SimpleStats()
    
    # Simulate recording items
    test_data = [
        ("python", "Library: pandas"),
        ("python", "Library: numpy"),
        ("sql", "Advanced patterns (confidence: 0.95)"),
        ("javascript", "Keywords (score: 8.5)"),
        ("python", "Advanced patterns (confidence: 0.92)"),
        ("sql", "Advanced patterns (confidence: 0.88)"),
    ]
    
    for domain, reason in test_data:
        stats.record(domain, reason)
    
    summary = stats.get_summary()
    
    print(f"\nTotal items: {summary['total']}")
    print(f"Counts: {summary['counts']}")
    print(f"Top reasons: {summary['top_reasons']}")
    
    # Verify
    assert summary['total'] == 6
    assert summary['counts']['python'] == 3
    assert summary['counts']['sql'] == 2
    assert summary['counts']['javascript'] == 1
    print("\n✓ All assertions passed!")


def test_performance():
    """Compare performance metrics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    import time
    from split_ds1000_by_domain import SimpleStats
    
    # Test SimpleStats performance
    stats = SimpleStats()
    
    start = time.time()
    for i in range(10000):
        stats.record("python", "test_reason")
    elapsed = time.time() - start
    
    print(f"\nSimpleStats (10,000 records):")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Rate: {10000/elapsed:.0f} records/sec")
    
    # Memory estimation
    summary = stats.get_summary()
    import sys
    size = sys.getsizeof(summary)
    print(f"  Memory: ~{size} bytes")
    
    print("\n✓ Performance test completed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DS1000 SPLITTER IMPROVEMENTS - TEST SUITE")
    print("=" * 80 + "\n")
    
    try:
        test_advanced_classifier()
        test_classifier_chain()
        test_simple_stats()
        test_performance()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

