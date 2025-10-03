python scraper_modal_retry.py \
  --time-range "Last 30 Days" \
  --data-type "Orders" \
  --expected-header "order_id,sku,qty,price" \
  --popup-action accept \
  --popup-timeout 15 \
  --confirm-selector ".modal-footer .btn-primary"
