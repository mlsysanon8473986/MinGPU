"""The method of the paper, one subpackage per stage of its workflow.

``define_search_space``
    Defines the search space: from the request, the candidate models, the GPU counts and the
    restrictions on the parallel plans.
``search``
    Evaluates each pair with Calculon (the inner search) and stores it.
``objectives``
    Chooses the answer among the stored pairs, under Objective 1 or Objective 2.
``workflow``
    Runs the three stages for one request.
"""
