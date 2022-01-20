//Remaps scalar float value from one range to another.
float Remap( float value, float low1, float high1, float low2, float high2 )
{
    return low2 + ( value - low1 ) * ( high2 - low2 ) / ( high1 - low1 );
}

//Remaps vector float values from one range to another.
float4 Remap( float4 value, float4 low1, float4 high1, float4 low2, float4 high2 )
{
    return low2 + ( value - low1 ) * ( high2 - low2 ) / ( high1 - low1 );
}


//Returns average luminosity of color, also known as grayscale.
float Luminosity( float3 color )
{
    return dot( float3(0.30, 0.59, 0.11), color );
}

float Luminosity( float4 color )
{
    return dot( float3(0.30, 0.59, 0.11), color.rgb );
}


//Converts color from RGB to HSV
float3 RGBtoHSV(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return  float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 HSVtoRGB(float3 c)
{
    float4 K    = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p    = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return      c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}



//!--UNTESTED--!!
//Packing and unpacking vector3 values to float32.
float Pack3PNForFP32(float3 channel)
{
    int uValue;
    uValue = ((int)(channel.x * 65535.0f + 0.5f));
    uValue |= ((int)(channel.y * 255.0f + 0.5f)) << 16;
    uValue |= ((int)(channel.z * 253.0f + 1.5f)) << 24;
    return (float)(uValue);
}

float3 Unpack3PNFromFP32(float fFloatFromFP32)
{
    float a, b, c, d;
    int uValue;
    int uInputFloat = (int)(fFloatFromFP32);
    a = ((uInputFloat) & 0xFFFF) / 65535.0f;
    b = ((uInputFloat >> 16) & 0xFF) / 255.0f;
    c = (((uInputFloat >> 24) & 0xFF) - 1.0f) / 253.0f;
    return new float3(a, b, c);
}