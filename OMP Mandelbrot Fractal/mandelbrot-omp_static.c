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
#include <omp.h>

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

void color(int red, int green, int blue)
{
    fputc((char)red, stdout);
    fputc((char)green, stdout);
    fputc((char)blue, stdout);
}

int main(int argc, char *argv[])
{
    int w = 6000, h = 4000, x, y;
    
    /* each iteration, it calculates: newz = oldz*oldz + p, where p is the 
       current pixel, and oldz stars at the origin */
    double zoom = 1, moveX = -0.5, moveY = 0; /* you can change these to zoom and change position */
    int maxIterations = 10000;               /* function stops after maxIterations */
    
    clock_t begin, end;
    double time_spent;
    
    printf("P6\n# CREATOR: Eric R. Weeks / mandel program\n");
    printf("%d %d\n255\n",w,h);
    
    begin = omp_get_wtime();
    int row[w];
    /* loop through every pixel */
    for(y = 0; y < h; y++) {
        #pragma omp parallel
        {
            double pr, pi;                            /* real and imaginary part of the pixel p    */
            double newRe, newIm, oldRe, oldIm;
            #pragma omp for schedule(static)
            for(x = 0; x < w; x++) {
                /* 'i' will represent the number of iterations */
                /* calculate the initial real and imaginary part of z, based on the
                pixel location and zoom and position values */
                int i;
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
                row[x] = 0;
                if (i != maxIterations) {
                    double z = sqrt(newRe * newRe + newIm * newIm);
                    row[x] = 256 * log2(1.75 + i - log2(log2(z))) / log2((double)maxIterations);
                }
            }
        }
        int i;
        for (i=0; i < w; i++) {
            if (row[i] == 0) 
                color(0, 0, 0);
            else 
                color(row[i], row[i], 255);
        }
    }
    
    end = omp_get_wtime();
    
    time_spent = (double)(end - begin);
    fprintf(stderr, "Elapsed time: %.2f seconds.\n", time_spent);
    return 0;
}
