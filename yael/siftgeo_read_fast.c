#include <stdio.h>
#include "mex.h"

#define SIFTGEO_SIZE      168
#define SIFTGEO_DIM_DES   128
#define SIFTGEO_DIM_META  9

void mexFunction(int nlhs, mxArray *plhs[], 
		 int nrhs, const mxArray *prhs[])
{

#ifdef HAVE_OCTAVE
#define mwSize long
#endif

  /* For the filename */
  mwSize buflen;
  

    char *fsiftgeo_name;
    FILE * fsiftgeo;
    long n;
    int i, j, ret;
    
    /* SIFT descriptors and associated meta-data */
    double * v;
    double * meta;
    unsigned char * fbuffer;
    
    /* check for proper number of arguments */
    if(nrhs!=1) 
      mexErrMsgTxt("One input required.");
    else if(nlhs > 2) 
      mexErrMsgTxt("Too many output arguments.");

    /* input must be a string */
    if (mxIsChar(prhs[0]) != 1)
      mexErrMsgTxt("Input must be a string.");

    /* input must be a row vector */
    if (mxGetM(prhs[0])!=1)
      mexErrMsgTxt("Input must be a row vector.");
    
    /* get the length of the input string */
    buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0])) + 1;

    /* copy the string data from prhs[0] into a C string input_ buf.    */
    fsiftgeo_name = mxArrayToString(prhs[0]);
    
    if(fsiftgeo_name == NULL) {
      mxFree(fsiftgeo_name);
      mexErrMsgTxt("Could not convert input to string.");
    }

    /* open the file for reading and retrieve it size */
    fsiftgeo = fopen (fsiftgeo_name, "r");
    if (!fsiftgeo) 
      mexErrMsgTxt("Could not open the input file");
    mxFree(fsiftgeo_name);

    fseek (fsiftgeo, 0, SEEK_END);
    n = ftell (fsiftgeo) / SIFTGEO_SIZE;
    fseek (fsiftgeo, 0, SEEK_SET);
 
    /* Read all the data using a single read function, and close the file */
    fbuffer = malloc (n * SIFTGEO_SIZE);
    ret = fread (fbuffer, sizeof (*fbuffer), n * SIFTGEO_SIZE, fsiftgeo);
    if (ret != n * SIFTGEO_SIZE)
      mexErrMsgTxt("Unable to read correctly from the input file");
    fclose (fsiftgeo);

    /* Allocate the output matrices */
    plhs[0] = mxCreateDoubleMatrix(SIFTGEO_DIM_DES, n, mxREAL);
    v = mxGetPr(plhs[0]);

    if (nlhs > 1) {
      plhs[1] = mxCreateDoubleMatrix(SIFTGEO_DIM_META, n, mxREAL);
      meta = mxGetPr(plhs[1]);
    }

    /* Copy the data from the buffer into these variables */
    for (i = 0 ; i < n ; i++) {
      
      for (j = 0 ; j < SIFTGEO_DIM_DES ; j++)
	v[j + SIFTGEO_DIM_DES * i] = fbuffer[i * SIFTGEO_SIZE + j
	  + SIFTGEO_DIM_META * sizeof (float) + sizeof (int)];
      /*	v[i + j * n] = fbuffer[i * SIFTGEO_SIZE + j
		+ SIFTGEO_DIM_META * sizeof (float) + sizeof (int)]; */

      if (nlhs > 1) {
	float * fbuf = (float *) (fbuffer + i * SIFTGEO_SIZE);
	for (j = 0 ; j < SIFTGEO_DIM_META ; j++)
	  meta[j + SIFTGEO_DIM_META * i] = fbuf[j];
      }
    }
    free (fbuffer);
}