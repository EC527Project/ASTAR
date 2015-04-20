// nvcc -arch=sm_21 -o cuda_dstar cuda_dstar.cu -lrt -lm

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAPROW 	5
#define MAPCOL 	5
#define MAX 	9999
#define STRCOL 	0
#define STRROW 	0
#define DSTCOL 	4
#define DSTROW 	4
#define BLOCK 	5

typedef struct {
  int key1; // min(g(x), rhs(x) + h(x))
  int key2; // min(g(x), rhs(x))
} keyvalue;

typedef struct {
  int x, y; // coordinate
  int px, py; // parents
  int g, h, rhs; // parameters
  keyvalue key; // keyvalue
  int type; // terrain: to be defined, i.e. ostacle / river / mountain / ground
  int inlist; // used in openlist
} point;

// Initiation can be done parallelly.
// Parallel A* Kernel
// In each kernel, a starting and destination point is given and the optimal path is calculate.
// Path should be stored in a list. In the meantime, cost between the two points should be stored.

__device__ void CalculateKey(point *p)
{
	point *t = p;
	int temp;
	temp = t->rhs + t->h;
	t->key.key1 = (t->g > temp) ? temp : t->g;
	t->key.key2 = (t->g > t->rhs) ? t->rhs : t->g;
	//printf("key %d\t%d\t%d\t%d\n", p.y, p.x, p.key.key1, p.key.key2);
}

__device__ void InitMap(point *map) // can be done in GPU
{
	int i, j;
	for (i = 0; i < MAPROW; i++)
		for (j = 0; j < MAPCOL; j++)
		{
			map[i*BLOCK+j].g = MAX; // set an extremely large number  
			map[i*BLOCK+j].h = (i > j) ? i : j; // set the origin point as starting point
			map[i*BLOCK+j].rhs = MAX; // set an extremely large number
			map[i*BLOCK+j].inlist = 0; // not in list
			map[i*BLOCK+j].key.key1 = MAX;
			map[i*BLOCK+j].key.key2 = MAX;
		}
}

__device__ void CalculateCost(point dst, point src, int *cst)
{
	int flag, cost;
	flag = dst.type * src.type;
	switch (flag)
	{
		case 1: cost = 1;  break; // ground to ground
		case 2: cost = 5;  break; // ground to water / water to ground
		case 3: cost = 10; break; // ground to mount / mount to ground
		case 4: cost = 2;  break; // water to water
		case 6: cost = 20; break; // mount to water / water to mount
		case 9: cost = 15; break; // mount to mount
		default: cost = MAX; break;
	}
	*cst = cost;
}

__device__ void UpdateVertex(point p, point *map)
{
	// Update RHS
	int i, j, temp, cost, row, col;
	int r = p.rhs; // set a large value at first
	for (i = -1; i < 2; i++)
	{
		row = p.y + i;
		if (row >=0 && row < MAPROW)
			for (j = -1; j < 2; j++)
			{
				col = p.x + j;
				if (col >=0 && col < MAPCOL)
				{
					if (i == 0 && j == 0) cost = 0;
					else CalculateCost(p, map[row*BLOCK+col], &cost);
					temp = map[row*BLOCK+col].g + cost;
					if (r >= temp) 
					{
						r = temp;
						if (col != p.x || row != p.y)
						{
							p.px = col;
							p.py = row;
						}
					}
				}
			}
	}
	p.rhs = r;
	if (p.rhs != p.g) 
	{
		p.inlist = 1;
		CalculateKey(&p);
	}
	else p.inlist = 0;
	map[p.y*BLOCK+p.x] = p;
	//printf("update debug: %d %d %d %d\n", p.y, p.x, p.py, p.px);
	//printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", p.y, p.x, p.g, p.rhs, p.key.key1, p.key.key2, p.inlist);
}

__device__ void FindKey(point *map, int *cx, int *ry)
{
	int i, j, x, y;
	point *data = map;
	int temp1 = MAX;
	int temp2 = MAX;
	x = 0;
	y = 0;
	for (i = 0; i < MAPROW; i++)
	{
		for (j = 0; j < MAPCOL; j++)
		{
			if (data[i*BLOCK+j].inlist == 1) // only points inlist will be compared
			{
				keyvalue kk = data[i*BLOCK+j].key;
				if (temp1 > kk.key1) 
				{
					temp1 = kk.key1;
					temp2 = kk.key2;
					x = j;
					y = i;
				}
				else if (temp1 == kk.key1)
				{
					if (temp2 > kk.key2)
					{
						temp2 = kk.key2;
						x = j;
						y = i;
					}
				}
			}
		}
	}
	*cx = x;
	*ry = y;
	//printf("update: %d\t%d\n", y, x);
}

__device__ void ShortestPath(int cx, int ry, point *map)
{
	int position; // Top Key
	point *data = map;
	int i, j, row, col;
	point cur = data[ry*BLOCK+cx];
	//printf("debug: %d %d %d %d\n", cur.x, cur.y, cur.key.key1, cur.key.key2);
	if (cur.g > cur.rhs)
	{
		data[ry*BLOCK+cx].g = data[ry*BLOCK+cx].rhs;
		for (i = -1; i <= 1; i++)
		{
			row = ry + i;
			if (row >= 0 && row < MAPROW)
				for (j = -1; j <=1 ; j++)
				{
					col = cx + j;
					if (col >= 0 && col < MAPCOL)
					{
						//if (i != 0 || j!= 0)
						//{
						//printf("debug neighbors: %d %d\n", row, col);
							point temp = data[row*BLOCK+col];
							if (temp.type > 0) UpdateVertex(temp, map);
						//}
					}
				}
		}
	}
	else // update itself as well
	{
		data[ry*BLOCK+cx].g = MAX; // set a large value
		for (i = -1; i < 2; i++)
		{
			row = ry + i;
			if (row >= 0 && row < MAPROW)
				for (j = -1; j < 2; j++)
				{
					col = cx + j;
					if (col >= 0 && col < MAPCOL)
					{
						point temp = data[row*BLOCK+col];
						if (temp.type > 0) UpdateVertex(temp, map);
					}
				}
		}
	}
}

// cuda kernel

__global__ void kernel_path(point *map, int mx, int my, int sx, int sy, int dx, int dy)
{
	point top = map[dy*BLOCK+dx];
	point goal = map[sy*BLOCK+sx];
	int row, col, i, j, it, ry, cx;
	int flag = 0;
	InitMap(map);
	cx = dx;
	ry = dy;
	map[ry*BLOCK+cx].inlist = 1; // put start point into list
	map[ry*BLOCK+cx].rhs = 0; // set start point rhs as 0
	map[ry*BLOCK+cx].key.key1 = map[(my-1)*BLOCK+mx-1].h;
	map[ry*BLOCK+cx].key.key2 = 0;
	keyvalue topkey = top.key; 
	keyvalue goalkey = goal.key;
	printf("kernel test: %d\n", map[0].key.key1);
	if (topkey.key1 < goalkey.key1 || (topkey.key1 == goalkey.key1 && topkey.key2 < goalkey.key2) || goal.rhs != goal.g)  flag = 1;
	else flag = 0;
	it = 0;
	while (flag)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
	{
		printf("iter %d\t%d\t%d\n", it, ry, cx);
		ShortestPath(cx, ry, map);
		//printf("dst inlist %d\n", map[MAPROW-1][MAPCOL-1].inlist);
		FindKey(map, &cx, &ry);
		//printf("iter %d\t%d\t%d\n", it, ry, cx);
		top = map[ry*BLOCK+cx];
		goal = map[STRROW*BLOCK+STRCOL];
		topkey = top.key;
		goalkey = goal.key;
		if (topkey.key1 < goalkey.key1 || (topkey.key1 == goalkey.key1 && topkey.key2 < goalkey.key2) || goal.rhs != goal.g)  flag = 1;
		else flag = 0;
		it++;
	}
}

int main()
{
	int i, j, row, col;
	point *map, *d_map;
	const size_t map_size = sizeof(point) * MAPROW * MAPCOL;
	cudaMallocHost((void**)&map, map_size);
	FILE *fp = fopen("map2.csv", "r");
	if (fp != NULL)
	{
		for (i = 0; i < MAPROW; i++)
			for (j = 0; j < MAPCOL; j++)
			{
				fscanf(fp, "%d,%d,%d\n", &map[i*BLOCK+j].y, &map[i*BLOCK+j].x, &map[i*BLOCK+j].type);
				//printf("%d\t%d\t%d\n", map[i][j].x, map[i][j].y, map[i][j].type);
			}
	}
	fclose(fp);
	cudaMalloc((void **)&d_map, map_size);
	cudaMemcpy(d_map, map, map_size, cudaMemcpyHostToDevice);
	printf("cpu test: %d\n", map[0].type);
	dim3 gridSize(1, 1, 1);
	//dim3 blockSize();
	kernel_path<<<gridSize, 1>>>(d_map, MAPCOL, MAPROW, STRCOL, STRROW, DSTCOL, DSTROW);
	cudaDeviceSynchronize();
	cudaMemcpy(map, d_map, map_size, cudaMemcpyDeviceToHost);
	// row = STRROW;
	// col = STRCOL;
	// point path = map[row*BLOCK+col];
	// while (row != DSTROW || col != DSTCOL)
	// {
	// 	printf("Coordinate[%d, %d]\n", row, col);
	// 	row = path.py;
	// 	col = path.px;
	// 	path = map[row*BLOCK+col];
	// }
	// printf("Coordinate[%d, %d]\n", row, col);
	cudaFreeHost(map);
	cudaFree(d_map);
}