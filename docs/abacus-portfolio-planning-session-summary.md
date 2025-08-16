# Abacus Portfolio Planning Session Summary

**Date**: August 16, 2025  
**Session Type**: Planning and Requirements Gathering  
**Objective**: Design implementation plan for model switching portfolio tracker

---

## Session Overview

This session focused on planning and designing a new portfolio tracking system called "naz100_sp500_abacus" that combines NAZ100 and SP500 stock universes using a model switching technique for monthly portfolio optimization.

## Key Accomplishments

### 1. Requirements Analysis
- **Identified Core Need**: Create a portfolio that dynamically switches between NAZ100 and SP500 stock universes based on monthly model performance analysis
- **Established Data Sources**: Leverage existing data at `/Users/donaldpg/pyTAAA_data/Naz100` and `/Users/donaldpg/pyTAAA_data/SP500`
- **Defined Integration Points**: Use existing `recommend_model.py` for switching decisions and `PyTAAA.py` for daily portfolio management

### 2. Architecture Design
- **Simplified Approach**: Decided on leveraging 90% of existing PyTAAA codebase rather than building from scratch
- **Data Strategy**: Create combined HDF5 file merging NAZ100 and SP500 stock histories
- **Minimal Code Changes**: Use wrapper scripts around existing `PyTAAA.py` and `recommend_model.py` functionality
- **Environment Variables**: Implement configuration through environment variables for maximum reusability

### 3. Implementation Strategy
- **Phased Approach**: Broke implementation into 6 logical phases with clear deliverables
- **Time Estimation**: Total estimated implementation time of 10 hours (2 hours per phase)
- **Testing Strategy**: Comprehensive testing at each phase before proceeding
- **Risk Mitigation**: Identified potential risks and established guardrails

## Planned Components

### New Scripts to Create
1. **`create_abacus_hdf5.py`** - Merge NAZ100 and SP500 data sources
2. **`daily_abacus_update.py`** - Daily portfolio update wrapper around PyTAAA.py
3. **`monthly_universe_evaluation.py`** - Monthly switching decision wrapper around recommend_model.py
4. **`generate_abacus_web.py`** - Enhanced web content generation
5. **`setup_abacus_portfolio.py`** - One-time system setup
6. **`run_abacus_daily.py`** - Main automation runner

### Existing Files to Modify (Minimal Changes)
1. **`PyTAAA.py`** - Add environment variable support for data paths
2. **`recommend_model.py`** - Add universe comparison mode
3. **`pytaaa_model_switching_params.json`** - Add abacus model path

### Data Structure Design
```
/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/
├── data_store/           # Portfolio performance files
├── quotes/              # Combined HDF5 data
├── symbols/             # Combined symbol lists
├── pyTAAA_web/         # Web output files
└── config/             # Configuration and history
```

## Technical Approach

### 1. Data Integration
- **HDF5 Merging**: Combine existing NAZ100 and SP500 HDF5 files while handling duplicates
- **Symbol Management**: Create unified symbol lists from both universes
- **Date Alignment**: Ensure consistent date ranges across merged data

### 2. Portfolio Management
- **Daily Updates**: Use existing PyTAAA.py infrastructure with environment variable configuration
- **Monthly Evaluation**: Leverage recommend_model.py to compare NAZ100 vs SP500 performance
- **Switching Logic**: Implement decision tracking and universe selection based on performance metrics

### 3. Web Integration
- **Existing Infrastructure**: Reuse existing web generation patterns
- **Enhanced Content**: Add switching timeline, universe comparison charts, and Monte Carlo plots
- **Plot Integration**: Copy existing recommendation_plot.png and model performance plots

## Implementation Plan Details

### Phase Breakdown
1. **Phase 1**: Data Infrastructure Setup (2 hours)
2. **Phase 2**: Configuration System (1.5 hours)
3. **Phase 3**: Daily Portfolio Management (2 hours)
4. **Phase 4**: Monthly Universe Evaluation (2 hours)
5. **Phase 5**: Integration and Automation (1.5 hours)
6. **Phase 6**: Validation and Documentation (1 hour)

### Quality Assurance
- **Comprehensive Testing**: Each phase must pass all tests before proceeding
- **Code Standards**: Follow existing PyTAAA patterns and PEP 8 guidelines
- **Error Handling**: Implement robust error handling and logging
- **Performance**: Minimize impact on existing systems

## Key Design Decisions

### 1. Simplification Over Complexity
- **Decision**: Use wrapper scripts instead of complex new architecture
- **Rationale**: Leverage proven, battle-tested existing code
- **Impact**: Faster implementation, lower maintenance burden

### 2. Combined Data Approach
- **Decision**: Create merged HDF5 file rather than dynamic data access
- **Rationale**: Simpler implementation, consistent with existing patterns
- **Impact**: One-time setup complexity, but ongoing simplicity

### 3. Environment Variable Configuration
- **Decision**: Use environment variables for runtime configuration
- **Rationale**: Minimal changes to existing code, maximum flexibility
- **Impact**: Clean separation between configuration and logic

## Success Metrics

### Technical Requirements
- [ ] Combined HDF5 file contains both universe data correctly
- [ ] Daily updates generate expected portfolio values
- [ ] Monthly evaluations make reasonable universe selections
- [ ] Web outputs render correctly with abacus-specific content
- [ ] Integration with existing PyTAAA systems works seamlessly

### Quality Metrics
- [ ] All tests pass with 100% success rate
- [ ] Code follows established PyTAAA patterns
- [ ] Error handling covers expected failure scenarios
- [ ] Performance impact on existing systems is minimal

## Risk Management

### Identified Risks and Mitigation Strategies
1. **Data Corruption**: Comprehensive validation and backup procedures
2. **Integration Breaking**: Extensive testing of existing workflows
3. **Performance Impact**: Benchmarking and optimization
4. **Configuration Complexity**: Simple defaults and clear documentation

## Next Steps

### Immediate Actions
1. **Review Implementation Plan**: Validate approach with stakeholders
2. **Set Up Development Environment**: Ensure all prerequisites are met
3. **Begin Phase 1**: Start with data infrastructure setup
4. **Establish Testing Framework**: Set up pytest structure for validation

### Implementation Approach
- **Agentic Development**: Use plan with Claude Sonnet 4 for efficient implementation
- **Incremental Progress**: Complete each phase fully before proceeding
- **Continuous Testing**: Validate each component as it's built
- **Documentation**: Maintain clear documentation throughout

## Files Created This Session

### Planning Documentation
- **`.github/implementation-plan-abacus-model-switching-instructions.md`** - Comprehensive implementation plan with task breakdown, time estimates, and checkboxes for progress tracking

## Session Outcomes

### 1. Clear Vision
- Established clear requirements and scope for the abacus portfolio system
- Defined integration points with existing PyTAAA infrastructure
- Created realistic timeline and resource estimates

### 2. Actionable Plan
- Detailed implementation plan with specific tasks and time estimates
- Clear phase structure with testing requirements
- Comprehensive risk assessment and mitigation strategies

### 3. Minimal Complexity
- Simplified architecture leveraging existing proven components
- Reduced development risk through maximum code reuse
- Clear separation of concerns with modular design

### 4. Implementation Ready
- All prerequisites identified and documented
- Development guidelines and standards established
- Testing strategy and success criteria defined

## Conclusion

This planning session successfully established a comprehensive, actionable plan for implementing the abacus portfolio tracking system. The approach prioritizes simplicity, code reuse, and integration with existing PyTAAA infrastructure while providing the desired model switching functionality. The plan is structured for efficient implementation with clear milestones and quality gates.

The next phase involves beginning implementation according to the established plan, starting with Phase 1: Data Infrastructure Setup.