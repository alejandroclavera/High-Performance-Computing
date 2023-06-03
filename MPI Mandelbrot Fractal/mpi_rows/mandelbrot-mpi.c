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


int* calculate_portion(int y_init, int y_end, int w, int h) 
{
    double pr, pi;                         
    double newRe, newIm, oldRe, oldIm;  
    int count = y_end - y_init;   
    fprintf(stderr, "start=%d, end=%d, count=%d\n", y_init, y_end, count);
    int *brightness = malloc(w * count * sizeof(int));
    int x,y; 
    
    for (y = y_init; y < y_end; y++) 
    {
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
            int idx = (w * (y-y_init)) + x;
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
    return brightness;
}

void print_fractal(int n_procs, int n_rows_proc, int last_proc_residual, int w) 
{
    MPI_Status status;
    int *brightness_buffer = malloc((n_rows_proc + last_proc_residual) * w * sizeof(int));
    int i,j;
    for (i=1; i < n_procs; i++) 
    {
        int count = w * (n_rows_proc + ((i == n_procs - 1) ? last_proc_residual : 0));
        fprintf(stderr, "count=%d\n", count);
        // wait to the calculator proc ends to read the calculated pixels
        MPI_Recv(brightness_buffer, count, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
        for (j=0; j<count; j++) 
        {
            int b = brightness_buffer[j];
            if (b==0)
                color(0, 0, 0);
            else 
                color(b, b, 255);      
        }
    }
}

int main(int argc, char *argv[])
{
    int w = 600, h = 400, x, y;
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
    int residual = h % n_operators_proc; 
    int n_rows_proc = h / n_operators_proc;
    int y_init = n_rows_proc * (iproc - 1);
    int y_end = y_init + n_rows_proc + ((iproc == nproc - 1) ? residual : 0);
    if (iproc != 0) 
    {
        int *brightness = calculate_portion(y_init, y_end, w, h); 
        MPI_Send(brightness, (y_end - y_init) * w, MPI_INT, 0, 0, MPI_COMM_WORLD);           
    } else 
    {
        print_fractal(nproc, n_rows_proc, residual, w);
    }
    end = MPI_Wtime();
    MPI_Finalize(); 

    if (iproc == 0) {
        time_spent = (double)(end - begin);
        fprintf(stderr, "Elapsed time: %.2f seconds.\n", time_spent);
    }
    return 0;
}
