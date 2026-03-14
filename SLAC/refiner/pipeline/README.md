# Refiner Pipeline

This directory contains the backend pipeline for:
raw file -> text extraction/cleaning -> rule-based chunk0 -> refiner input -> refiner inference -> refined chunks

Current goal:
1. unify file reading
2. stabilize rule-based chunk0 generation
3. convert chunk0 to refiner-standard atoms+b0 input
4. run frozen Week 2 refiner checkpoint
5. export refined_chunks.jsonl for retrieval
