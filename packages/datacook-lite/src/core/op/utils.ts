import { CUR_BACKEND } from "../../env";
export const getMethodErrorStr = (methodName: string): string => {
  return `Method ${methodName} cannot found in current backend ${CUR_BACKEND}`;
};
