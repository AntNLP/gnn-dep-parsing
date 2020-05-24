# GNN Dependency Parser
The code of "Graph-based Dependency Parsing with Graph Neural Networks".



## Requirements

* `python`: 3.6.0
* [`dynet`](https://dynet.readthedocs.io/en/latest/python.html): 2.0.0
* [`antu`](https://github.com/JT-Ushio/antu): 0.0.5


## Example log

An example of [experiment log](https://github.com/JT-Ushio/gnn-dep-parsing-1/blob/master/gnn-dep-parser/exper.log).

|  PTB-UAS   | PTB-LAS  |
|  ----  | ----  |
| 96.0455  | 94.3539 |



## Training

```bash
$ cd src
$ python train.py --config_file ../configs/default.cfg --name ACL19(your experiment name) --gpu 0(your gpu id)
```
Before triggering the subcommands, please make sure that the data files must be in [CoNLL-U](https://universaldependencies.org/format.html) format. Here is an example.

```bash
$ cat data/dev.debug 
1	Influential	_	JJ	JJ	_	2	amod	_	_
2	members	_	NNS	NNS	_	10	nsubj	_	_
3	of	_	IN	IN	_	2	prep	_	_
4	the	_	DT	DT	_	6	det	_	_
5	House	_	NNP	NNP	_	6	nn	_	_
6	Ways	_	NNP	NNP	_	3	pobj	_	_
7	and	_	CC	CC	_	6	cc	_	_
8	Means	_	NNP	NNP	_	9	nn	_	_
9	Committee	_	NNP	NNP	_	6	conj	_	_
10	introduced	_	VBD	VBD	_	0	root	_	_
11	legislation	_	NN	NN	_	10	dobj	_	_
12	that	_	WDT	WDT	_	14	nsubj	_	_
13	would	_	MD	MD	_	14	aux	_	_
14	restrict	_	VB	VB	_	11	rcmod	_	_
15	how	_	WRB	WRB	_	22	advmod	_	_
16	the	_	DT	DT	_	20	det	_	_
17	new	_	JJ	JJ	_	20	amod	_	_
18	savings-and-loan	_	NN	JJ	_	20	nn	_	_
19	bailout	_	NN	NN	_	20	nn	_	_
20	agency	_	NN	NN	_	22	nsubj	_	_
21	can	_	MD	MD	_	22	aux	_	_
22	raise	_	VB	VB	_	14	ccomp	_	_
23	capital	_	NN	NN	_	22	dobj	_	_
24	,	_	,	,	_	14	punct	_	_
25	creating	_	VBG	VBG	_	14	xcomp	_	_
26	another	_	DT	DT	_	28	det	_	_
27	potential	_	JJ	JJ	_	28	amod	_	_
28	obstacle	_	NN	NN	_	25	dobj	_	_
29	to	_	TO	TO	_	28	prep	_	_
30	the	_	DT	DT	_	31	det	_	_
31	government	_	NN	NN	_	33	poss	_	_
32	's	_	POS	POS	_	31	possessive	_	_
33	sale	_	NN	NN	_	29	pobj	_	_
34	of	_	IN	IN	_	33	prep	_	_
35	sick	_	JJ	JJ	_	36	amod	_	_
36	thrifts	_	NNS	NNS	_	34	pobj	_	_
37	.	_	.	.	_	10	punct	_	_
```



## Predict

```bash
$ cd src
$ python predict.py --config_file ../configs/default.cfg --name PTB-Out(your experiment name) --gpu 0(your gpu id)
```



## Cite

If you find our code is useful, please cite:

```
@inproceedings{ji-etal-2019-graph,
    title = "Graph-based Dependency Parsing with Graph Neural Networks",
    author = "Ji, Tao  and
      Wu, Yuanbin  and
      Lan, Man",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1237",
    doi = "10.18653/v1/P19-1237",
    pages = "2475--2485",
}
```

