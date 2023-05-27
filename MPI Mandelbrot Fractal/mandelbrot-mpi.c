/*
 * mandelbrot-seq.c
 * 
 *
 * The Mandelbrot calculation is to iterate the equation
 * z = z*z + c, where z and c are complex numbers, z is initially
 * zero, and c is the coordinate of the point being tested. If
 * the magnitude of z remains less than 2 for ever, then the point
 * c is in the Mandelbrot set. In this code We write out the number of iterations
 * before the magnitude of z exceeds 2, or UCHAR_MAX, whichever is
 * smaller.
*/

#include <stdio.h>
#include<mpi.h>

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>



double zoom = 1, moveX = -0.5, moveY = 0; /* you can change these to zoom and change position */
int maxIterations = 10000;  


void color(int red, int green, int blue)
{
    fputc((char)red, stdout);
    fputc((char)green, stdout);
    fputc((char)blue, stdout);
}


void calculate_portion(int y, int xInit, int xEnd, int h, int w, int *brightness) {
    double pr, pi;                         
    double newRe, newIm, oldRe, oldIm;  
    int x;   
    for(x = xInit; x < xEnd; x++) {
        /* 'i' will represent the number of iterations */
        int i;
        
        /* calculate the initial real and imaginary part of z, based on the
            pixel location and zoom and position values */
        pr = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
        pi = (y - h / 2) / (0.5 * zoom * h) + moveY;
        newRe = newIm = oldRe = oldIm = 0.0; /* these should start at 0.0 */
        
        /* start the iteration process */
        for(i = 0; i < maxIterations; i++) {
            /* remember value of previous iteration */
            oldRe = newRe;
            oldIm = newIm;
            /* the actual iteration, the real and imaginary part are calculated */
            newRe = oldRe * oldRe - oldIm * oldIm + pr;
            newIm = 2 * oldRe * oldIm + pi;
            /* if the point is outside the circle with radius 2: stop */
            if((newRe * newRe + newIm * newIm) > 4) break;
        }
        
        /* color(i % 256, 255, 255 * (i < maxIterations)); */
        if(i == maxIterations) {
            brightness[x - xInit] = 0;
        } else {
            double z = sqrt(newRe * newRe + newIm * newIm);
            int currentBrightness = 256 * log2(1.75 + i - log2(log2(z))) / log2((double)maxIterations);
            brightness[x - xInit] = currentBrightness;
        }
    }
}

int main(int argc, char *argv[])
{
    int w = 600, h = 400, x, y;
    int iproc, nproc;
    
    MPI_Status status;
    clock_t begin, end;
    double time_spent;
    
    MPI_Init( &argc, &argv ); 
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);    
    if (iproc == 0) {
        printf("P6\n# CREATOR: Eric R. Weeks / mandel program\n");
        printf("%d %d\n255\n",w,h);
    }

    begin = clock();
    int brightness[w];
    // calculate load distribution
    int n_operators_proc = nproc - 1;
    int residual = w % n_operators_proc; 
    int n_cols_proc = residual + w / n_operators_proc;
    int x_init = n_cols_proc * (iproc - 1);
    int x_end = (x_init + n_cols_proc);

    fprintf(stderr, "irpoc=%d s=%d end=%d \n", iproc, x_init, x_end);
    /* loop through every pixel */
    for(y = 0; y < h; y++) {
        if (iproc != 0) {
            calculate_portion(y, x_init, x_end, h, w, brightness);            
            MPI_Send(brightness, n_cols_proc, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else {
            int i,j;
            for (i=1; i < nproc; i++) {
                int count = (n_cols_proc + residual);
                MPI_Recv(brightness, count, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
                for (j=0; j<count; j++) {
                    int b = brightness[j];
                    fprintf(stderr, "%d ", b);
                    if (b==0)
                        color(0, 0, 0);
                    else
                        color(b, b, 255);
                }
            }
            fprintf(stderr, "\n");
        }
    }
    MPI_Finalize(); 
    
    end = clock();
    
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    fprintf(stderr, "Elapsed time: %.2f seconds.\n", time_spent);
    
    return 0;
}
