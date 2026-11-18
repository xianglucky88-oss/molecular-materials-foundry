import math

def orientation(positions):
    a,b,c,d=positions
    rows=[[p[i]-d[i] for i in range(3)] for p in (a,b,c)]
    determinant=(rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1])
                 -rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0])
                 +rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]))
    scale=math.prod(math.sqrt(math.fsum(x*x for x in row)) for row in rows)
    if scale==0 or abs(determinant)<=1e-10*scale:
        return None
    return "R" if determinant<0 else "S"


def contains_center(center, positions):
    """The central atom must lie strictly inside the four-ligand tetrahedron."""
    def determinant(rows):
        a,b,c=rows
        return (a[0]*(b[1]*c[2]-b[2]*c[1])
                -a[1]*(b[0]*c[2]-b[2]*c[0])
                +a[2]*(b[0]*c[1]-b[1]*c[0]))
    for opposite in range(4):
        face=[positions[i] for i in range(4) if i!=opposite]
        origin=face[0]
        edges=[[face[j][k]-origin[k] for k in range(3)] for j in (1,2)]
        relative=[center[k]-origin[k] for k in range(3)]
        vertex=[positions[opposite][k]-origin[k] for k in range(3)]
        side=determinant(edges+[relative])
        reference=determinant(edges+[vertex])
        if reference==0 or side*reference<=0 or abs(side/reference)<=1e-10:
            return False
    return True
