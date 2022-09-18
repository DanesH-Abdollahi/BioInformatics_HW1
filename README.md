# Semi-Global alignment

### Some Examples :
Input Format :
``` python
>sequence1
>sequence2
```
Output Format :
``` python
Alignment Score : int
Alignment :
    sequence1
    sequence2
```

---
Input 1 :
```python
HEAGAWGHE
PAWHEA
```
Output 1 :
```python
20
HEAGAWGHE-
---PAW-HEA
```
---

---
Input 2 :
``` python
AAAAA
AA
```
Output 2 :
``` python
4
AAAAA
---AA
AAAAA
--AA-
AAAAA
-AA--
AAAAA
AA---
```
---
