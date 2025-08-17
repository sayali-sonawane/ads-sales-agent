"""
Tests for the main module.
"""

import pytest
from sales_agent.main import main


def test_main_function():
    """
    Test that the main function runs without errors.
    """
    # This is a basic test to ensure the main function can be called
    # In a real application, you would test actual functionality
    try:
        main()
        assert True  # If we get here, the function ran without errors
    except Exception as e:
        pytest.fail(f"Main function raised an exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
