"""
@Problem: 2D Cook's membrane. A benchmark example.

@Formulation: Mixed displacement-pressure formulation.

@Material: Incompressible Linear elastic.

@author: Dr Chennakesava Kadapa

Created on Sun 18-Jun-2024
"""


# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# For MPI-based parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# PETSc solvers
from petsc4py import PETSc

# specific functions from dolfinx modules
from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
#from dolfinx.io import VTXWriter

# specific functions from ufl modules
import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, sym, det, div, dev, inv, tr, sqrt, conditional ,\
                 gt, dx, ds, inner, derivative, dot, ln, split, exp, eq, cos, acos, ge, le)

# basix finite elements
import basix
from basix.ufl import element, mixed_element, quadrature_element




msh, cell_tags, facet_tags = io.gmshio.read_from_msh("petals-4leaves-size0p1-Q2.msh", MPI.COMM_WORLD)

# coordinates of the nodes
x = ufl.SpatialCoordinate(msh)


# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx = ufl.Measure('dx', domain=msh, subdomain_data=cell_tags, metadata={'quadrature_degree': 4})

#dx4 = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 4}, subdomain_id=4)
#dx5 = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 4}, subdomain_id=5)

ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 4})



# FE Elements
# Quadratic element for displacement
###
###
# Define function spaces and elements
deg_u = 2
deg_p = deg_u-1

# displacement
U2 = element("Lagrange", msh.basix_cell(), deg_u, shape=(3,))
# pressure
P1 = element("Lagrange", msh.basix_cell(), deg_p)

# Mixed element
TH = mixed_element([U2, P1])
ME = functionspace(msh, TH)


V2 = functionspace(msh, U2) # Vector function space


# functions with DOFs at the current step
w = Function(ME)
u, p = split(w)

# functions with DOFs at the previous step
w_old = Function(ME)
u_old, p_old = split(w_old)

w_old2 = Function(ME)
u_old2, p_old2 = split(w_old2)

# Test functions
u_test, p_test = TestFunctions(ME)


# Trial functions
dw = TrialFunction(ME)


# Boundary conditions
'''
1 1 "corneredge"
2 2 "symX"
2 3 "symY"
3 4 "growth"
3 5 "nongrowth"
'''

dofs_2_x = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(2)) #ux
bc2_x = fem.dirichletbc(value=0.0, dofs=dofs_2_x, V=ME.sub(0).sub(0))

dofs_2_z = fem.locate_dofs_topological(ME.sub(0).sub(2), facet_tags.dim, facet_tags.find(2)) #ux
bc2_z = fem.dirichletbc(value=0.0, dofs=dofs_2_z, V=ME.sub(0).sub(2))


dofs_3_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(3)) #uy
bc3_y = fem.dirichletbc(value=0.0, dofs=dofs_3_y, V=ME.sub(0).sub(1))


def Corner(x):
    return np.logical_and( np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)), np.isclose(x[2],-0.02) )


#corner_facets = mesh.locate_entities_boundary(msh, 0, Corner)

#corner_dofs = fem.locate_dofs_geometrical(V2, Corner)
#corner_dofs = fem.locate_dofs_geometrical(ME.sub(0).sub(0), Corner)

#corner_dofs = fem.locate_dofs_topological(ME.sub(0).sub(0), 1, facet_tags.find(1)) #ux

V0, submap = ME.sub(0).sub(0).collapse()
corner_dofs_x = fem.locate_dofs_geometrical((ME.sub(0).sub(0),V0), Corner)

V0, submap = ME.sub(0).sub(1).collapse()
corner_dofs_y = fem.locate_dofs_geometrical((ME.sub(0).sub(1),V0), Corner)

V0, submap = ME.sub(0).sub(2).collapse()
corner_dofs_z = fem.locate_dofs_geometrical((ME.sub(0).sub(2),V0), Corner)


#bc1_x = fem.dirichletbc(value=0.0, dofs=corner_dofs_x[0], V=ME.sub(0).sub(0))
#bc1_y = fem.dirichletbc(value=0.0, dofs=corner_dofs_y[0], V=ME.sub(0).sub(1))
bc1_z = fem.dirichletbc(value=0.0, dofs=corner_dofs_z[0], V=ME.sub(0).sub(2))

#bcs = [bc1_x, bc1_y, bc1_z, bc2_x, bc3_y]
bcs = [bc1_z, bc2_x, bc3_y]
#bcs = [bc2_x, bc3_y, bc2_z]


# Parameter values
G_f = Constant(msh, 1000.0)
G_s = Constant(msh, 10000.0)

kappa_inv = Constant(msh, PETSc.ScalarType(0.0))


t = 0.0
dt = 0.001
time_final = 0.1

# dimension
d = len(u)


# Kinematics
Id = Identity(3)
F  = ufl.variable(Id + grad(u))
J  = ufl.variable(det(F))
C  = ufl.variable(F.T*F)


## growth part, say it is the film
growthf  = Constant(msh, t)
g11f = growthf
g22f = growthf
g33f = growthf

Fgf = ufl.as_tensor( ((1.0+g11f,0.0,0.0),(0.0,1.0+g22f,0.0),(0.0,0.0,1.0+g33f)) )

FgfInv = ufl.variable(inv(Fgf))
Jgf = ufl.variable(det(Fgf))
Fef = ufl.variable(F*FgfInv)
Jef = ufl.variable(det(Fef))
Cef = ufl.variable(Fef.T*Fef)

PK1_f   = Jef**(-2/3)*G_f*(Fef - 1/3*tr(Cef)*inv(Fef.T)) + Jef*p*inv(Fef.T)

# Weak form
Res_f   = inner(PK1_f,   grad(u_test))*Jgf*dx(4) + inner((Jef-1-p*kappa_inv), p_test)*Jgf*dx(4)


## non-growth part, say it is the substrate
PK1_s   = J**(-2/3)*G_s*(F - 1/3*tr(C)*inv(F.T)) + J*p*inv(F.T)

# Weak form
Res_s   = inner(PK1_s,   grad(u_test))*dx(5) + inner((J-1-p*kappa_inv), p_test)*dx(5)

Res = Res_f + Res_s


dRes = derivative(Res, w, dw)


# set up the nonlinear problem
problem = NonlinearProblem(Res, w, bcs, dRes)



# set the solver parameters
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.report = True


#  The Krylov solver parameters.
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()


# displacement projection
U1 = element("Lagrange", msh.basix_cell(), 1)

Vs = functionspace(msh, U2)

u_proj = Function(V2)
u_proj.name = "displacement"

p_proj = Function(functionspace(msh, U1))
p_proj.name = "pressure"


fname = "petals-disp-.pvd"
VTKfile_Disp = io.VTKFile(msh.comm, fname, "w")
VTKfile_Disp.write_mesh(msh)

fname = "petals-pres-.pvd"
VTKfile_Pres = io.VTKFile(msh.comm, fname, "w")
VTKfile_Pres.write_mesh(msh)

# function to write results to XDMF at time t
def writeResults(time_cur, timeStep):

    # Displacement
    u_proj.interpolate(w.sub(0))
    VTKfile_Disp.write_function(u_proj)

    # Displacement
    p_proj.interpolate(w.sub(1))
    VTKfile_Pres.write_function(p_proj)



print("\n----------------------------\n")
print("Simulation has started")
print("\n----------------------------\n")

timeStep = 0
time_cur = 0.0

writeResults(time_cur, timeStep)


# loop over time steps
#for timeStep in range(num_steps):
while ( round(time_cur+dt, 6) <= time_final):
    # Update current time
    time_cur += dt
    timeStep += 1

    print("\n\n Load step = ", timeStep)
    print("     Time      = ", time_cur)

    growthf.value = time_cur
    #growths.value = 0.0
    
    w.x.array[:] = 2*w_old.x.array[:] - w_old2.x.array[:]

    # Solve the problem
    # Compute solution
    (num_its, converged) = solver.solve(w)
    
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged.")

    writeResults(time_cur, timeStep)

    # save variables
    # DOFs
    w_old2.x.array[:] = w_old.x.array[:]
    w_old.x.array[:] = w.x.array[:]
    #w.vector.copy(w_old.vector)Disp
    #velocity
    #v_temp.interpolate(v_expr)
    #v_old.x.array[:] = v_temp.x.array[:]
    #acceleration
    #a_temp.interpolate(a_expr)
    #a_old.x.array[:] = a_temp.x.array[:]


VTKfile_Disp.close()
VTKfile_Pres.close()

