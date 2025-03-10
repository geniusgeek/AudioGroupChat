import type { Headers } from "../types";
export type SortDirection = "asc" | "des";
export declare function get_sort_status(name: string, sort_by: number | undefined, direction: SortDirection | undefined, headers: Headers): "none" | "ascending" | "descending";
export declare function sort_data(data: {
    id: string;
    value: string | number;
}[][], sort_by: number | undefined, sort_direction: SortDirection | undefined): number[];
