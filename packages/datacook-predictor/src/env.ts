export const getEnv = (value?: string): string => {
  if(value) {
    Object.assign(global, {
      ...global,
      ENV: value
    })
  }
  const envProp = Object.getOwnPropertyDescriptor(global, 'ENV');
  if(envProp) {
    return envProp.value.toString();
  } else {
    return 'cpu';
  }
};
