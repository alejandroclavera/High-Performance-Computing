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


void operator_process(int start, int increment, int w, int h) 
{ 
    int *brightness = malloc(w * sizeof(int));
    int y;
    for (y = start; y < h; y+=increment) 
    {
         #pragma omp parallel
        {
                double pr, pi;                         
                double newRe, newIm, oldRe, oldIm;  
                int x;
                #pragma omp for schedule(static)
                for(x = 0; x < w; x++) 
                {
                    /* 'i' will represent the number of iterations */
                    int i;
                    
                    /* calculate the initial real and imaginary part of z, based on the
                        pixel location and zoom and position values */
                    pr = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
                    pi = (y - h / 2) / (0.5 * zoom * h) + moveY;
                    newRe = newIm = oldRe = oldIm = 0.0; /* these should start at 0.0 */
                    
                    /* start the iteration process */
                    for(i = 0; i < maxIterations; i++) 
                    {
                        /* remember value of previous iteration */
                        oldRe = newRe;
                        oldIm = newIm;
                        /* the actual iteration, the real and imaginary part are calculated */
                        newRe = oldRe * oldRe - oldIm * oldIm + pr;
                        newIm = 2 * oldRe * oldIm + pi;
                        /* if the point is outside the circle with radius 2: stop */
                        if((newRe * newRe + newIm * newIm) > 4) break;
                    }
                    int idx = x;
                    if(i == maxIterations) 
                    {
                        brightness[idx] = 0;
                    } else 
                    {
                        double z = sqrt(newRe * newRe + newIm * newIm);
                        int currentBrightness = 256 * log2(1.75 + i - log2(log2(z))) / log2((double)maxIterations);
                        brightness[idx] = currentBrightness;
                    }
                }
        }
        MPI_Send(brightness, w, MPI_INT, 0, 0, MPI_COMM_WORLD);           
    }
}

void master_process(int n_procs, int w, int h) 
{
    MPI_Status status;
    int *brightness_buffer = malloc(n_procs * w * sizeof(int));
    int proc,j;
    int total_rows = 0;
    int windows_start = 0;
    while(total_rows < h) {
        int total_in_window = 0;
        for (proc = 0; proc < n_procs; proc++) 
        {
            // ignore procs without work
            if (windows_start + proc >= h) 
                continue;
            // wait to the proc row
            int offset = w * proc;
            MPI_Recv(brightness_buffer + offset, w, MPI_INT, proc + 1, 0, MPI_COMM_WORLD, &status);
            total_rows += 1;
            total_in_window +=1;
           
        }
         // print the rows
        for (j=0; j< total_in_window * w; j++) 
        {
            int b = brightness_buffer[j];
            if (b==0)
                color(0, 0, 0);
            else 
                color(b, b, 255);      
        }
        windows_start += n_procs;
    }
}

int main(int argc, char *argv[])
{
    int w = 6000, h = 4000, x, y;
    clock_t begin, end;
    double time_spent;
    int iproc, nproc;

    MPI_Init( &argc, &argv ); 
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc); 

    // Print image header   
    if (iproc == 0) 
    {
        printf("P6\n# CREATOR: Eric R. Weeks / mandel program\n");
        printf("%d %d\n255\n",w,h);
    }

    // start the fractal generation
    begin = MPI_Wtime();
    // calculate load distribution
    int n_operators_proc = nproc - 1;
    if (iproc != 0) 
    {
        operator_process(iproc - 1, n_operators_proc, w, h); 
    } else 
    {
        master_process(n_operators_proc, w, h);
    }
    end = MPI_Wtime();
    MPI_Finalize(); 

    if (iproc == 0) {
        time_spent = (double)(end - begin);
        fprintf(stderr, "Elapsed time: %.2f seconds.\n", time_spent);
    }
    return 0;
}
