# babygrad Tutorial Guidelines

## Target Audience

The reader:
- Is familiar with Python and software engineering patterns
- Is **new to deep learning concepts** (tensors, autograd, neural networks)
- Appreciates things being made **clear and explicit** rather than assumed
- Wants to **implement the code themselves**, not copy solutions

## Pedagogical Approach

### Explain concepts before code

Before showing any code, explain:
- **What** the thing is (in plain language)
- **Why** we need it (what problem does it solve?)
- **How** it connects to what we've already built

Use analogies when helpful (e.g., "Tensor is like a tracked package—the item plus its shipping history").

### Be explicit about Python/NumPy behavior

Don't assume the reader knows how library functions behave. For example:
- Show what `np.array()` accepts: lists, scalars, tuples, nested lists
- Explain what `isinstance()` does with examples
- Explain what `@property` does and why you'd use it

### Provide structure, not solutions

When creating exercises:
- Give the **skeleton code** with the control flow already in place (if/elif/else, loops, etc.)
- Use `_____` as fill-in-the-blank placeholders
- Add **inline comments** that guide toward the solution without giving it away
- Ask leading questions: "What attribute of a Tensor contains the numpy array?"

**Good example:**
```python
if isinstance(data, Tensor):
    # We need to extract the numpy array from inside it
    # A Tensor's numpy array is stored in its .data attribute
    data = _____  # What attribute of the input Tensor contains the numpy array?
```

**Bad example (too vague):**
```python
# if data isinstance of Tensor
# YOUR CODE HERE
```

**Bad example (gives away the answer):**
```python
if isinstance(data, Tensor):
    data = data.data
```

### Use hint boxes for supplementary info

Put background explanations in hint boxes so they don't interrupt the flow:
- Python concepts (`isinstance`, `@property`, `__repr__` vs `__str__`)
- NumPy behavior (`np.array()` flexibility, `.copy()` for independence)
- Design rationale (why return a copy? why default to True?)

### Connect chapters forward and backward

- End chapters with "What's Next?" previewing how this builds toward the next topic
- Reference earlier chapters when building on concepts
- Use consistent terminology across chapters

## File Organization

- `/docs/public/` — Tutorial HTML files
- `/babygrad/` — Actual implementation (student fills this in)
- `/tests/` — Tests to verify implementations
- `/examples/` — Working examples using babygrad

## Styling

- Mark revised chapters with the green "✓ Revised for clarity" badge
- Use tables for comparing cases/options
- Use `<div class="hint">` for supplementary explanations
- Use `<div class="exercise">` for exercise callouts
- Keep code blocks focused—one concept per block

## Content Revision Workflow

When revising a chapter:
1. Keep the **same information** as the original
2. Add **explanations** for concepts
3. Convert solutions to **fill-in-the-blank exercises**
4. Add **hints** that guide without giving answers
5. Test by asking: "Could a beginner follow this without googling?"

## Attribution

Every page links to the original zekcrates tutorial. Revised pages note "(revised for clarity)" in the attribution.
