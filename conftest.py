# conftest.py — sits in project root
# Makes 'src' importable when running scripts directly
import sys, os
sys.path.insert(0, os.path.abspath("."))