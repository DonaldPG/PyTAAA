# Constructive Critic Review Guide

**Purpose:** Review the refactoring plan for feasibility, completeness, and risk mitigation  
**Reviewer Role:** Experienced software architect, skeptical of AI-generated plans  
**Review Focus:** Identify gaps, unnecessary complexity, and risks

---

## Review Checklist

### 1. Phase Boundaries

- [ ] Are phases appropriately sized? (Can each be completed in 2-6 AI sessions?)
- [ ] Are dependencies between phases clear?
- [ ] Could any phases be combined without increasing risk?
- [ ] Should any phase be split due to complexity?

**Questions to Answer:**
1. Is Phase 4 (Separate Computation from I/O) too large? Should it be split into data loading and computation extraction?
2. Is Phase 5 (Break Up TAfunctions.py) appropriately sequenced after Phase 4?
3. Are there any hidden dependencies between phases not documented?

### 2. Test Coverage

- [ ] Does each phase have adequate test coverage?
- [ ] Are the tests actually validating behavioral equivalence?
- [ ] Are there edge cases not covered?

**Questions to Answer:**
1. Are the end-to-end tests (7 commands) sufficient to catch behavioral changes?
2. Should there be unit tests for specific functions in earlier phases?
3. Is the test for "exact output match" too strict? (timestamps in logs, etc.)

### 3. Risk Assessment

- [ ] Are risks appropriately identified for each phase?
- [ ] Are mitigation strategies adequate?
- [ ] What could go wrong that's not documented?

**Questions to Answer:**
1. What if the baseline capture itself has issues?
2. What if HDF5 data changes between baseline and validation runs?
3. What if exception handling changes expose a failure that was previously silently caught and now breaks the pipeline?

### 4. AI Model Recommendations

- [ ] Are model assignments appropriate for task complexity?
- [ ] Is cost optimization appropriate without sacrificing quality?
- [ ] Are there tasks where a different model would be better?

**Questions to Answer:**
1. Is Kimi K2.5 sufficient for Phase 1, or should Claude be used throughout for consistency?
2. Is o1 necessary for Phases 4-5, or would Claude 4.5 Sonnet be sufficient?
3. Should there be a "human review checkpoint" after Phase 2 before proceeding to higher-risk phases?

### 5. "AI Slop" Detection

- [ ] Are there unnecessary abstractions?
- [ ] Is there over-engineering?
- [ ] Are there patterns that look good but add complexity without value?

**Questions to Answer:**
1. Is the exception logging decorator (`exception_logger.py`) unnecessary complexity?
2. Is the module decomposition in Phase 5 creating too many small files?
3. Are the data loader abstractions in Phase 4 adding indirection without benefit?

### 6. Operational Considerations

- [ ] Will daily/monthly workflows be disrupted?
- [ ] Is the rollback plan practical?
- [ ] Are there operational risks not addressed?

**Questions to Answer:**
1. What if a phase introduces a bug that's only caught during the monthly model recommendation run?
2. Should there be a "canary" period where both old and new code run side-by-side?
3. Is the git commit strategy appropriate for a trading system where stability is critical?

### 7. Missing Elements

- [ ] What's missing from the plan?
- [ ] Are there prerequisites not addressed?
- [ ] Are there post-phase activities not documented?

**Questions to Answer:**
1. Should there be a performance benchmarking phase?
2. Should there be a security review of the refactored code?
3. Should there be documentation updates as part of each phase?

---

## Review Output Template

```markdown
## Constructive Critic Review: PyTAAA Refactoring Plan

**Reviewer:** [AI Model or Human Name]  
**Date:** [Review Date]  
**Plan Version:** 1.0

### Overall Assessment

[GO / NO-GO / GO WITH MODIFICATIONS]

### Strengths

1. [Strength 1]
2. [Strength 2]
3. [Strength 3]

### Concerns

#### High Priority

1. **[Concern Title]**
   - **Issue:** [Description]
   - **Impact:** [What could go wrong]
   - **Recommendation:** [How to address]

#### Medium Priority

1. **[Concern Title]**
   - **Issue:** [Description]
   - **Impact:** [What could go wrong]
   - **Recommendation:** [How to address]

#### Low Priority

1. **[Concern Title]**
   - **Issue:** [Description]
   - **Impact:** [What could go wrong]
   - **Recommendation:** [How to address]

### Specific Recommendations

#### Phase-Specific Changes

**Phase 1:**
- [Recommendation 1]
- [Recommendation 2]

**Phase 2:**
- [Recommendation 1]
- [Recommendation 2]

[Continue for all phases]

#### General Recommendations

1. [Recommendation that applies to multiple phases or overall plan]
2. [Another general recommendation]

### AI Model Recommendation Changes

| Phase | Current | Recommended | Rationale |
|-------|---------|-------------|-----------|
| [N] | [Current Model] | [New Model] | [Why change] |

### Additional Considerations

[Anything else the reviewer wants to highlight]

### Final Verdict

[Summary of whether the plan is ready for human review and execution]
```

---

## Review Process

1. **Initial Review:** AI constructive critic reviews the plan using this guide
2. **Feedback Integration:** Architect addresses feedback, updates plan
3. **Re-review (if needed):** Critic reviews updated plan
4. **Human Review:** Human reviews final plan and critic feedback
5. **Approval:** Human approves plan for execution

---

## Example Review Snippets

### Example: High Priority Concern

```markdown
#### High Priority

1. **Phase 2 Exception Handling Risk**
   - **Issue:** The two-step approach (logging mode then fix mode) doubles the work 
     and may not catch all exception types in the logging phase if test coverage 
     doesn't trigger all code paths.
   - **Impact:** Exceptions that only occur in production (network timeouts, 
     specific data edge cases) won't be observed during testing, leading to 
     incomplete exception specifications.
   - **Recommendation:** Add a "safety fallback" where the specific except clause 
     also catches Exception and logs unexpected types for future refinement:
     ```python
     try:
         risky_operation()
     except (ExpectedError1, ExpectedError2) as e:
         handle_expected(e)
     except Exception as e:
         logger.error(f"Unexpected exception type {type(e)}: {e}")
         raise  # Re-raise to maintain existing behavior
     ```
```

### Example: AI Slop Detection

```markdown
### "AI Slop" Detection

1. **Exception Logger Decorator**
   - **Issue:** The `exception_logger.py` module creates a new abstraction that 
     is only used temporarily during Phase 2.
   - **Impact:** Adds code that will be deleted, increasing churn.
   - **Recommendation:** Inline the logging logic directly in the except blocks 
     during the logging phase, then remove it in the fix phase. Don't create 
     a separate module for temporary code.
```

---

**End of Review Guide**
