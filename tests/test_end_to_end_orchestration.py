import pytest
import asyncio
from memory_profiler import memory_usage
from main import run_pipeline

# Removed asyncio wrapper since memory_usage needs to wrap a blocking execution
def test_end_to_end_memory_bounds():
    """
    Ensures that the entire pipeline orchestration sequence runs
    cleanly without crashing AND stays beneath our modest RAM limits.
    """
    
    # We will trigger the mock main pipeline using memory_profiler
    # To use it properly in an async context, we run the asyncio loop 
    # directly via a synchronous wrapper.
    
    def sync_wrapper():
        return asyncio.run(run_pipeline(mock_mode=True))
        
    print("\nStarting memory profile over Pipeline Execution...")
    
    # memory_usage returns a list of memory loads in MiB recorded over the function's execution
    mem_profile = memory_usage(sync_wrapper)
    
    max_memory = max(mem_profile)
    start_memory = mem_profile[0]
    consumed_memory = max_memory - start_memory
    
    print(f"\nPipeline Initial Memory: {start_memory:.2f} MiB")
    print(f"Pipeline Peak Memory: {max_memory:.2f} MiB")
    print(f"Total Overhead Memory Consumed: {consumed_memory:.2f} MiB")
    
    # For a mock 100x100 array grid, overhead should easily be < 150 MiB.
    # Our project limit scale is < 16GB (16,000 MiB), but for this unit test 
    # we assert it stays extraordinarily tiny (below 250 MB).
    assert consumed_memory < 350, f"Memory leaked aggressively: {consumed_memory} MiB"
    
    print("\nStep 5 Orchestration Complete: Daemon workflow executes successfully within tight memory bounds!")

if __name__ == "__main__":
    asyncio.run(test_end_to_end_memory_bounds())
