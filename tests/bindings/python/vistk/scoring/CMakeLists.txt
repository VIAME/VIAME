project(vistk_test_python_vistk_scoring)

##############################
# Scoring result tests
##############################
vistk_build_python_test(scoring_result test-scoring_result.py)

vistk_make_python_test(scoring_result import)
vistk_make_python_test(scoring_result api_calls)
