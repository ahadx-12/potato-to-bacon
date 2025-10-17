@equation(domain="classical")
def kinetic_energy(m: Mass(kg), v: Speed(m_per_s)) -> Energy(J):
    return 0.5*m*v**2
