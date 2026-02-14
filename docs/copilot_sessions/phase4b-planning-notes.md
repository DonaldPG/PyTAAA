# Phase 4b Planning Notes

**Status:** Ready to begin (separate session recommended)  
**Complexity:** High  
**Risk:** High (architectural changes)  
**Estimated Time:** 3-4 hours

## Overview

Phase 4b continues the testability improvements begun in Phase 4a by extracting plotting and file I/O from `PortfolioPerformanceCalcs()`. The goal is to create a pure computation function that can be unit tested without side effects.

## Current State (After Phase 4a)

✅ Data loading extracted to `functions/data_loaders.py`  
✅ PortfolioPerformanceCalcs.py uses extracted loader  
🔄 Still contains: plotting, file writing, print statements, computation logic (all mixed together)

## Phase 4b Goals

1. Extract plot generation from `PortfolioPerformanceCalcs()`
2. Extract file writing from `PortfolioPerformanceCalcs()`
3. Create pure `compute_portfolio_metrics()` function with no side effects
4. Refactor `PortfolioPerformanceCalcs()` as orchestrator that calls: load → compute → output

## Complexity Assessment

**Why High Risk:**
- PortfolioPerformanceCalcs.py is 639 lines with deeply intertwined logic
- Many side effects: plots, file writes, print statements throughout
- Heavy use of numpy operations that must preserve exact semantics
- Multiple call sites that depend on current behavior
- Unclear separation between computation and output in some sections

**Key Challenges:**
1. Identifying which variables are computation vs output
2. Determining minimal pure function interface
3. Maintaining exact numerical behavior (floating point)
4. Preserving all side effects (plots, files, prints) in correct order
5. Handling error cases and edge conditions

## Recommended Approach

### Pre-Work (Before Starting 4b)

1. **Code Reading Session:**
   - Read PortfolioPerformanceCalcs.py lines 1-639 carefully
   - Map out all side effects (file writes, plots, prints)
   - Identify computation blocks vs I/O blocks
   - Document data flow through the function

2. **Dependency Analysis:**
   - Use grep to find all calls to PortfolioPerformanceCalcs
   - Check what return values are actually used
   - Verify backward compatibility requirements

3. **Test Strategy Design:**
   - Plan shadow tests similar to Phase 4a
   - Identify golden outputs for comparison
   - Consider property-based tests for computation invariants

### Implementation Strategy (When Ready)

**Option A: Conservative (Recommended)**
- Extract one small piece at a time (e.g., just plot generation first)
- Shadow test each extraction separately
- Commit incrementally (4b1, 4b2, 4b3, etc.)
- Full validation after each sub-phase

**Option B: Bold**
- Extract everything in one go
- Extensive shadow testing at end
- Single large commit
- Higher risk, faster if successful

**Recommendation:** Use Option A. PortfolioPerformanceCalcs is too complex to refactor in one step safely.

### Sub-Phase Breakdown (Option A)

**Phase 4b1: Extract Plot Generation (Lowest Risk)**
- Create `functions/output_generators.py`
- Move plot generation calls to `generate_plots()` function
- Pass computed results as parameters
- Shadow test: Verify plots identical

**Phase 4b2: Extract File Writing (Medium Risk)**
- Add `write_output_files()` to output_generators.py
- Move all file writes (.params files, etc.)
- Shadow test: Verify file contents identical

**Phase 4b3: Create Pure Computation Function (Highest Risk)**
- Create `compute_portfolio_metrics()` in PortfolioPerformanceCalcs.py
- Extract all computation into pure function
- No I/O, no side effects, just math
- Shadow test: Verify results identical
- Property tests: Verify mathematical invariants

**Phase 4b4: Orchestration Refactor**
- Rewrite PortfolioPerformanceCalcs() as orchestrator:
  ```python
  def PortfolioPerformanceCalcs(symbol_directory, symbol_file, params, json_fn):
      # Load
      adjClose, symbols, datearray = load_quotes_for_analysis(...)
      
      # Compute
      results = compute_portfolio_metrics(adjClose, symbols, datearray, params)
      
      # Output
      generate_plots(results, params, output_dir)
      write_output_files(results, params, output_dir)
      
      return results['last_date'], results['symbols'], results['weights'], results['prices']
  ```
- Shadow test: End-to-end verification

### Testing Strategy

**Required Tests:**
1. Shadow tests for each sub-phase (compare old vs new)
2. Property tests for computation function (mathematical invariants)
3. Unit tests for pure functions with synthetic data
4. Integration tests for orchestrator
5. Full e2e validation (all 7 commands)

**Test Data:**
- Use static data from `/Users/donaldpg/pyTAAA_data_static/`
- Capture baseline outputs before starting
- Compare outputs after each sub-phase

### Risk Mitigation

**Before Each Sub-Phase:**
- [ ] Full test suite passing
- [ ] E2E baseline captured
- [ ] Git committed to safe point
- [ ] Plan rollback strategy

**During Each Sub-Phase:**
- [ ] Shadow tests passing before moving on
- [ ] No logic changes, pure extraction
- [ ] Preserve exact semantics (even quirks/bugs)
- [ ] Document any surprising discoveries

**After Each Sub-Phase:**
- [ ] Full test suite passing
- [ ] E2E validation identical to baseline
- [ ] Git commit with descriptive message
- [ ] Update refactoring plan

### Success Criteria

**Phase 4b is complete when:**
1. ✅ Pure `compute_portfolio_metrics()` function exists
2. ✅ All plots generated by `generate_plots()`
3. ✅ All files written by `write_output_files()`
4. ✅ PortfolioPerformanceCalcs() is thin orchestrator
5. ✅ All 106+ tests passing
6. ✅ All 7 e2e commands produce identical outputs
7. ✅ No performance regression (within 10%)
8. ✅ Comprehensive shadow tests for all extractions

### Time Estimates

- Pre-work (code reading, analysis): 1 hour
- Phase 4b1 (plot extraction): 1 hour
- Phase 4b2 (file extraction): 1 hour
- Phase 4b3 (pure computation): 2 hours (most complex)
- Phase 4b4 (orchestration): 30 minutes
- Testing & validation: 1 hour
- **Total: 6.5 hours** across multiple sessions

### Red Flags (Stop and Reassess If...)

⚠️ Shadow tests show differences in outputs  
⚠️ Test suite time increases by >20%  
⚠️ New bugs discovered in existing code  
⚠️ Unclear how to extract a section cleanly  
⚠️ Multiple failed attempts at same extraction  

**If red flag encountered:** Stop, document issue, get human review before proceeding.

## Open Questions (To Resolve Before Starting)

1. **Return value structure:** What does `compute_portfolio_metrics()` return?
   - Single dict? Multiple dicts? Dataclass? Named tuple?
   - Trade-offs: dict is flexible but untyped, dataclass is typed but rigid

2. **Error handling:** Where should errors be caught?
   - In pure function? In orchestrator? In I/O functions?
   - Current code has try/except blocks scattered throughout

3. **Print statements:** Keep, remove, or convert to logging?
   - Many print statements provide user feedback
   - Some are debugging statements
   - Should pure function be silent?

4. **Side effects in computation:** Are there any?
   - Does computation modify global state?
   - Does it write temporary files?
   - Does it cache results?

5. **Performance implications:** Will extraction add overhead?
   - Function call overhead negligible
   - Data copying might be issue if large arrays
   - Profile before/after to verify

## Resources

- **Refactoring Plan:** `plans/REFACTORING_PLAN_final.md` Phase 4b (lines 1329-1530)
- **Target File:** `functions/PortfolioPerformanceCalcs.py` (639 lines)
- **Phase 4a Session:** `docs/copilot_sessions/2026-02-13_phase4a-data-loading-extraction.md`
- **Static Test Data:** `/Users/donaldpg/pyTAAA_data_static/`

## Recommendation for Next Session

**Before starting Phase 4b implementation:**

1. Schedule dedicated 2-hour block for code reading
2. Create detailed map of PortfolioPerformanceCalcs.py:
   - Identify all side effects with line numbers
   - Map data flow through function
   - Document all return values and their usage
3. Write up findings in a design document
4. Review with human collaborator if possible
5. **Only then** begin implementation

**Do NOT rush into Phase 4b.** The complexity and risk warrant careful planning. Phase 4a's success came from careful preparation and shadow testing. Phase 4b requires even more care.

---

**Status:** Planning complete, ready for implementation in dedicated future session
