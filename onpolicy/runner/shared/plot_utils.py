import matplotlib.pyplot as plt
import numpy as np
import os

def mkdir(task_id = None):

    task_id = "task" + str(task_id)
    if not os.path.exists(os.path.join('figures', task_id)):
        os.makedirs(os.path.join('figures', task_id))

    return os.path.join('figures', task_id)


def plot_quaternion(data_list, data_name):
    save_path = mkdir()
    time_list = np.array([i for i in range(len(data_list))])

    q_1_list = []
    q_2_list = []
    q_3_list = []
    q_4_list = []

    for data in data_list:
        q_1, q_2, q_3, q_4 = data
        q_1_list.append(q_1)
        q_2_list.append(q_2)
        q_3_list.append(q_3)
        q_4_list.append(q_4)

    plt.plot(time_list, q_1_list, label='$q_1$')
    plt.plot(time_list, q_2_list, label='$q_2$')
    plt.plot(time_list, q_3_list, label='$q_3$')
    plt.plot(time_list, q_4_list, label='$q_4$')

    plt.xlabel('Time' + ', s', fontsize=16)
    plt.ylabel(data_name, fontsize=16)

    plt.legend(loc='right', fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()



def plot_omega(data_list, data_name, data_unit='rad/s'):
    save_path = mkdir()
    time_list = np.array([i for i in range(len(data_list))])

    omega_x_list = []
    omega_y_list = []
    omega_z_list = []

    for data in data_list:
        omega_x, omega_y, omega_z = data
        omega_x_list.append(omega_x)
        omega_y_list.append(omega_y)
        omega_z_list.append(omega_z)



    # plt.plot(time_list, omega_x_list,linestyle = '-', label='omega_x')
    # plt.plot(time_list, omega_y_list,linestyle = '--', label='omega_y')
    # plt.plot(time_list, omega_z_list,linestyle = '-.', label='omega_z')

    plt.plot(time_list, omega_x_list, label='$\omega_x$')
    plt.plot(time_list, omega_y_list, label='$\omega_y$')
    plt.plot(time_list, omega_z_list, label='$\omega_z$')

    plt.ylim(-np.pi / 10, np.pi / 10)

    plt.xlabel('Time' + ', s', fontsize=16)
    plt.ylabel(data_name + ', ' + data_unit, fontsize=16)

    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()

def plot_detected_time(data_list, data_name='Tracking Time', data_unit='s'):
    save_path = mkdir()
    time_list = np.array([i for i in range(len(data_list))])

    plt.plot(time_list, data_list, label=data_name)
    plt.xlabel('Time' + ', s', fontsize=16)
    plt.ylabel(data_name + ', ' + data_unit, fontsize=16)

    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()

def plot_angle(data_list, data_name='Angle', data_unit='deg', for_target = True):
    save_path = mkdir()
    time_list = np.array([i for i in range(len(data_list))])

    plt.plot(time_list, data_list, label='Angle')

    if for_target:
        # for target
        plt.axhline(y = METADATA['half_angle'], color='red', linestyle=':')
        plt.text( 100, METADATA['half_angle'] + 2, 'Payload Half Angle', color='red', rotation=0, ha='center', va='bottom')
    else:
        # for forbidden zones
        plt.axhline(y = METADATA['forbidden_half_angle'], color='red', linestyle=':')
        plt.text( 100, METADATA['forbidden_half_angle'] + 2, 'Forbidden Half Angle', color='red', rotation=0, ha='center', va='bottom')

    plt.xlabel('Time' + ', s', fontsize=16)
    plt.ylabel(data_name + ', ' + data_unit, fontsize=16)
    plt.ylim(bottom=0)

    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()

def plot_angles(angles_list, data_name='Angle', data_unit='deg'):
    save_path = mkdir()

    # 1 target + N_zone Zones
    for a in range( 1 + METADATA['N_zone']):
        data_list = angles_list[a]
        time_list = np.array([i for i in range(len(data_list))])

        if a == 0:
            label = 'Angle (Target)'
        else:
            label = 'Angle (Forbidden Zone{})'.format(a)
        plt.plot(time_list, data_list, label = label)


    plt.axhline(y = METADATA['half_angle'], color='red', linestyle=':')
    plt.text( 200, METADATA['half_angle'] + 1, 'Payload Half Angle', color='red', rotation=0, ha='center', va='bottom', fontsize = 10)

    plt.axhline(y = METADATA['forbidden_half_angle'], color='red', linestyle=':')
    plt.text( 200, METADATA['forbidden_half_angle'] - 1, 'Forbidden Half Angle', color='red', rotation=0, ha='center', va='top', fontsize = 10)

    plt.xlabel('Time' + ', s', fontsize=16)
    plt.ylabel('Angle, ' + data_unit, fontsize=16)
    # plt.ylim(bottom=0)
    plt.ylim(0, 140)

    plt.legend(loc='upper right', fontsize = 8)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()


def plot_torque(data_list, data_name='Actual Torque', data_unit='Nm'):
    save_path = mkdir()
    time_list = np.array([i for i in range(len(data_list))])

    torque_1_list = []
    torque_2_list = []
    torque_3_list = []
    # torque_4_list = []

    for data in data_list:
        # torque_1, torque_2, torque_3, torque_4 = data
        torque_1, torque_2, torque_3 = data
        torque_1_list.append(torque_1)
        torque_2_list.append(torque_2)
        torque_3_list.append(torque_3)
        # torque_4_list.append(torque_4)

    # plt.plot(time_list, torque_1_list, label='$torque_1$')
    # plt.plot(time_list, torque_2_list, label='$torque_2$')
    # plt.plot(time_list, torque_3_list, label='$torque_3$')
    # plt.plot(time_list, torque_4_list, label='$torque_4$')

    plt.plot(time_list, torque_1_list, label='$u_x$')
    plt.plot(time_list, torque_2_list, label='$u_y$')
    plt.plot(time_list, torque_3_list, label='$u_z$')

    # plt.ylim(METADATA['torque_range'][0] * 1.5, METADATA['torque_range'][1] * 1.5)

    plt.xlabel('Time' + ', s', fontsize=16)
    plt.ylabel(data_name + ', ' + data_unit, fontsize=16)

    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()


