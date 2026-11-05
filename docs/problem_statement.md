# Problem statement: EV charging stations with station types and substation capacity

## Summary

Decide where to install EV charging stations in a city represented as a graph, and choose in each location between a **slow station** (cheaper, lower capacity, smaller coverage radius) or a **fast station** (more expensive, higher capacity, larger radius). Each candidate is connected to a neighbourhood electrical substation, and the total installed capacity attached to each substation cannot exceed what that grid can supply.

**Metric to optimise:** maximise the total city demand effectively served.

**Constraints:** total budget; capacity of every installed station; a zone can only be served by a station within its coverage radius (which depends on the chosen type); electrical capacity of every substation, considering all stations connected to it.

## Formal formulation

### Sets
- $V$: graph nodes (street intersections). $E$: graph edges (street segments).
- $J \subseteq V$: candidate nodes where a station can be installed.
- $T=\{\text{slow},\text{fast}\}$: station types.
- $K$: electrical substations; $g(j)\in K$ is the substation of candidate $j$.

### Parameters
- $l_e$: length of edge $e\in E$. $d_G(i,j)$: shortest-path distance on $G=(V,E)$ (Dijkstra).
- $w_i$: estimated charging demand at node $i$.
- $c_j^t$: cost of installing a type-$t$ station at $j$. $cap_j^t$: maximum demand it can serve.
- $R^t$: coverage radius of type $t$ ($R^{fast}>R^{slow}$).
- $B$: total budget. $Cap_k$: maximum capacity substation $k$ can supply.

### Decision variables
- $x_j^t\in\{0,1\}$: a type-$t$ station is installed at $j$ (at most one type per candidate).
- $y_{ij}^t\ge 0$ for $d_G(i,j)\le R^t$: demand of node $i$ served by the type-$t$ station at $j$.

### Objective
$$\max_{x,y}\; F(x,y)=\sum_{i\in V}\sum_{j\in J}\sum_{t\in T} y_{ij}^t$$

### Constraints
1. $\sum_{t} x_j^t \le 1 \quad \forall j\in J$ — at most one station type per candidate.
2. $\sum_{j}\sum_{t} c_j^t x_j^t \le B$ — budget.
3. $\sum_{i} y_{ij}^t \le cap_j^t\, x_j^t \quad \forall j,t$ — station capacity.
4. $\sum_{j}\sum_{t} y_{ij}^t \le w_i \quad \forall i$ — served demand cannot exceed real demand.
5. $y_{ij}^t = 0$ if $d_G(i,j)>R^t$ — coverage by station type.
6. $\sum_{j:\,g(j)=k}\sum_{t} cap_j^t x_j^t \le Cap_k \quad \forall k\in K$ — substation capacity.
7. $y_{ij}^t\ge 0$.

*(Translated from the course's original Spanish problem statement, "Problema actualizado: con tipos de estación y capacidad de subestación eléctrica".)*
