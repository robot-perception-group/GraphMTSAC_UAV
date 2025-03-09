#pragma once

#include "FIFOBuffer.h"

#include "AC_CustomControl_config.h"
#include "AC_CustomControl_Backend.h"
#include <vector>

class AC_CustomControl_XYZ : public AC_CustomControl_Backend
{
private:        
    FIFOBuffer fifoBuffer;
    std::vector<float> env_latent;
    std::vector<float> NN_out;

    void resetAdaptorBuffer();
    void updateNNInput(const Quaternion& attitude_body, const Quaternion& attitude_target, const Vector3f& gyro_latest, const Vector3f& airspeed_earth_ned);
    void handleSpoolState();

public:
    // Mark var_info as static so it can be referenced externally
    static const AP_Param::GroupInfo var_info[];  // <--- ADD THIS AS STATIC

    // Constructor
    AC_CustomControl_XYZ(AC_CustomControl &frontend, AP_AHRS_View *&ahrs,
                         AC_AttitudeControl *&att_control,
                         AP_MotorsMulticopter *&motors, float dt);

    Vector3f update(void) override;
    void reset(void) override;
    std::vector<float> forward_policy(const std::vector<float>& state, const std::vector<float>& env_latent);
    std::vector<float> forward_adaptor(void);

protected:
    AP_Float authority;
};
