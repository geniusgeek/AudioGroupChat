import type { CellCoordinate, EditingState } from "./types";
export type CellData = {
    id: string;
    value: string | number;
};
export declare function is_cell_selected(cell: CellCoordinate, selected_cells: CellCoordinate[]): string;
export declare function get_range_selection(start: CellCoordinate, end: CellCoordinate): CellCoordinate[];
export declare function handle_selection(current: CellCoordinate, selected_cells: CellCoordinate[], event: {
    shiftKey: boolean;
    metaKey: boolean;
    ctrlKey: boolean;
}): CellCoordinate[];
export declare function handle_delete_key(data: CellData[][], selected_cells: CellCoordinate[]): CellData[][];
export declare function handle_editing_state(current: CellCoordinate, editing: EditingState, selected_cells: CellCoordinate[], editable: boolean): EditingState;
export declare function should_show_cell_menu(cell: CellCoordinate, selected_cells: CellCoordinate[], editable: boolean): boolean;
export declare function get_next_cell_coordinates(current: CellCoordinate, data: CellData[][], shift_key: boolean): CellCoordinate | false;
export declare function move_cursor(key: "ArrowRight" | "ArrowLeft" | "ArrowDown" | "ArrowUp", current_coords: CellCoordinate, data: CellData[][]): CellCoordinate | false;
export declare function get_current_indices(id: string, data: CellData[][]): [number, number];
export declare function handle_click_outside(event: Event, parent: HTMLElement): boolean;
export declare function select_column(data: any[][], col: number): CellCoordinate[];
export declare function select_row(data: any[][], row: number): CellCoordinate[];
export declare function calculate_selection_positions(selected: CellCoordinate, data: {
    id: string;
    value: string | number;
}[][], els: Record<string, {
    cell: HTMLTableCellElement | null;
}>, parent: HTMLElement, table: HTMLElement): {
    col_pos: string;
    row_pos: string | undefined;
};
