export declare function toggle_header_menu(event: MouseEvent, col: number, active_header_menu: {
    col: number;
    x: number;
    y: number;
} | null, set_active_header_menu: (menu: {
    col: number;
    x: number;
    y: number;
} | null) => void): void;
export declare function toggle_cell_menu(event: MouseEvent, row: number, col: number, active_cell_menu: {
    row: number;
    col: number;
    x: number;
    y: number;
} | null, set_active_cell_menu: (menu: {
    row: number;
    col: number;
    x: number;
    y: number;
} | null) => void): void;
export declare function add_row_at(index: number, position: "above" | "below", add_row: (index?: number) => void, clear_menus: () => void): void;
export declare function add_col_at(index: number, position: "left" | "right", add_col: (index?: number) => void, clear_menus: () => void): void;
export declare function delete_row_at(index: number, delete_row: (index: number) => void, clear_menus: () => void): void;
export declare function delete_col_at(index: number, delete_col: (index: number) => void, clear_menus: () => void): void;
export declare function toggle_header_button(col: number, active_button: {
    type: "header" | "cell";
    row?: number;
    col: number;
} | null, set_active_button: (button: {
    type: "header" | "cell";
    row?: number;
    col: number;
} | null) => void): void;
export declare function toggle_cell_button(row: number, col: number, active_button: {
    type: "header" | "cell";
    row?: number;
    col: number;
} | null, set_active_button: (button: {
    type: "header" | "cell";
    row?: number;
    col: number;
} | null) => void): void;
