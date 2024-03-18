#include<stdio.h>
#include<limits.h>
int main()
{
    int nm,nt;
    printf("\nenter no.of tasks and machines:\n");
    scanf("%d %d",&nm,&nt);
    int maxmin[nm][nt];
    int temp[nm][nt];
    int makespan=0;
    printf("\n fill data\n");
    for(int i=0;i<nm;i++)
    {
        for(int j=0;j<nt;j++)
        {
            scanf("%d",&maxmin[i][j]);
            temp[i][j]=maxmin[i][j];

        }
    }
    printf("\n original data\n");
    for(int i=0;i<nm;i++)
    {
        for(int j=0;j<nt;j++)
        
            printf("%d ",maxmin[i][j]);
        printf("\n");
        
    }
    int resulttask[nt];
    int resultmachine[nt];
    int resulttime[nt];
    int ptr=-1;
    while(ptr<nt-1)
    {
        
        int time[nt],machine[nt];
        for(int i=0;i<nt;i++)
        {
            int minimum=INT_MAX;
            int pos=-1;
            for(int j=0;j<nm;j++)
            {
                if(maxmin[i][j]<minimum){
                    minimum=maxmin[i][j];
                    pos=j;
                }
            }
            time[i]=minimum;
            machine[i]=pos;
        }
        int maximum=INT_MIN;
        int pos=-1;
        for(int i=0;i<nt;i<i++)
        {
            if(time[i]>maximum&&time[i]!=INT_MAX)
            {
                maximum=time[i];
                pos=i;
            }
        }
        resulttask[++ptr]=pos;
        resultmachine[ptr]=machine[pos];
        resulttime[ptr]=temp[machine[pos]][pos];
        if(maximum>makespan)
        {
            makespan=maximum;
        }
        for(int i=0;i<nm;i++)
        {
            for(int j=0;j<nt;j++)
            {
                if(j==resulttask[ptr])
                {
                    maxmin[i][j]=INT_MAX;

                }
                else if(i==resultmachine[ptr]&& maxmin[i][j]!=INT_MAX)
                {
                    maxmin[i][j]+=maximum;

                }
                else{
                    continue;

                }
            }
        }
    }
    printf("\nscheduled tasks are:\n");
    
        for(int i=0;i<nt;i++)
        {
            printf("\ntask %d runs on machine %d with time %d units\n",resulttask[i]+1,resultmachine[i]+1,resulttime[i]);
        }
        printf("\nmakespan %d units\n",makespan);
    

}