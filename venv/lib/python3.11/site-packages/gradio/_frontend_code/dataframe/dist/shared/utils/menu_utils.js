export function toggle_header_menu(event, col, active_header_menu, set_active_header_menu) {
    event.stopPropagation();
    if (active_header_menu && active_header_menu.col === col) {
        set_active_header_menu(null);
    }
    else {
        const header = event.target.closest("th");
        if (header) {
            const rect = header.getBoundingClientRect();
            set_active_header_menu({ col, x: rect.right, y: rect.bottom });
        }
    }
}
export function toggle_cell_menu(event, row, col, active_cell_menu, set_active_cell_menu) {
    event.stopPropagation();
    if (active_cell_menu &&
        active_cell_menu.row === row &&
        active_cell_menu.col === col) {
        set_active_cell_menu(null);
    }
    else {
        const cell = event.target.closest("td");
        if (cell) {
            const rect = cell.getBoundingClientRect();
            set_active_cell_menu({ row, col, x: rect.right, y: rect.bottom });
        }
    }
}
export function add_row_at(index, position, add_row, clear_menus) {
    const row_index = position === "above" ? index : index + 1;
    add_row(row_index);
    clear_menus();
}
export function add_col_at(index, position, add_col, clear_menus) {
    const col_index = position === "left" ? index : index + 1;
    add_col(col_index);
    clear_menus();
}
export function delete_row_at(index, delete_row, clear_menus) {
    delete_row(index);
    clear_menus();
}
export function delete_col_at(index, delete_col, clear_menus) {
    delete_col(index);
    clear_menus();
}
export function toggle_header_button(col, active_button, set_active_button) {
    set_active_button(active_button?.type === "header" && active_button.col === col
        ? null
        : { type: "header", col });
}
export function toggle_cell_button(row, col, active_button, set_active_button) {
    set_active_button(active_button?.type === "cell" &&
        active_button.row === row &&
        active_button.col === col
        ? null
        : { type: "cell", row, col });
}
