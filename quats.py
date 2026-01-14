import numpy as np # Scientific computing library for Python
 
def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qw, qx, qy, qz]


# Conjugado de un cuaternion
def quat_inv(q: np.array) -> np.array:
   q_cnj = np.zeros(4)
   q_cnj[0] = q[0]
   q_cnj[1] = -q[1]
   q_cnj[2] = -q[2]
   q_cnj[3] = -q[3]
   return q_cnj

# Inversa de un cuaternion
def quat_inv(q: np.array) -> np.array:
  return quat_inv(q)/np.sum(np.power(q, 2))

# Multiplicar un quat
def quat_mul(q1: np.array, q2: np.array):
  w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
  w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
  ww = (z1 + x1) * (x2 + y2)
  yy = (w1 - y1) * (w2 + z2)
  zz = (w1 + y1) * (w2 - z2)
  xx = ww + yy + zz
  qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
  w = qq - ww + (z1 - y1) * (y2 - z2)
  x = qq - xx + (x1 + w1) * (x2 + w2)
  y = qq - yy + (w1 - x1) * (y2 + z2)
  z = qq - zz + (z1 + y1) * (w2 - x2)
  return np.ndarray([w, x, y, z])

def quat_apply(quat: np.array, vec: np.array):
  xyz= quat[1:]
  t = np.cross(xyz, vec) * 2
  return (vec + quat[:1]*t+ np.cross(xyz, t))

# Sacar la posición relativa
def sbs_frame_transform(t01: np.array, q01: np.array, t02: np.array,  q02: np.ndarray):
   # Calculo orientacion
   # Calcular la inversa de 1
  q10 = quat_inv(q01)
   # Multiplicar 10 y 02
  q12 = quat_mul(q10, q02)
   # Aplicar orientacion a la posicion
  t12 = quat_apply(q10, t02-t01)
   # Devolver la posicion y el cuaternión.
  return t12, q12
if __name__ == "__main__":
    print(get_quaternion_from_euler(0,0,90))