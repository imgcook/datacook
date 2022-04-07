// export const getEnv = (value?: string): string => {
//   if(value) {
//     Object.assign(global, {
//       ...global,
//       ENV: value
//     })
//   }
//   const envProp = Object.getOwnPropertyDescriptor(global, 'ENV');
//   if(envProp) {
//     return envProp.value.toString();
//   } else {
//     return 'cpu';
//   }
// };
export const CUR_BACKEND = process.env.BACKEND;
export const IS_CPU_BACKEND = process.env.BACKEND === 'cpu';
